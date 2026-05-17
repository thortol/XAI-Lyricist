from mido import MidiFile

import os
import sys
import urllib.request
import requests
import re, sys
import pyphen
import json


def convert_midi_to_notes(path_to_midi):
    midi = MidiFile(path_to_midi)

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']

    def midi_to_note(note_num):
        octave = (note_num // 12) - 1
        name = NOTE_NAMES[note_num % 12]
        return f"{name}{octave}"

    notes = {}
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000
    def convert_to_seconds(time):
        return (time * tempo) / (ticks_per_beat * 1_000_000)

    output_notes = []
    total_time = 0
    for track in midi.tracks:
        for msg in track:
            total_time += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            # note_on with velocity > 0 means a real note start
            if msg.type == "note_on" and msg.velocity > 0:
                notes[msg.note] = convert_to_seconds(total_time)
            if (msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)) and msg.note in notes:
                output_notes.append((midi_to_note(msg.note), notes[msg.note], convert_to_seconds(total_time)))
                del notes[msg.note]
    return output_notes, tempo


def renderize_voice(xml_data, out_folder=".", vibpower=1, f0shift=0, synalpha=0.55):
	return sinsy_request(xml_data, vibpower, f0shift, synalpha)

def sinsy_request(xml_data, vibpower=1, f0shift=0, synalpha=0.55):

	headers = {'User-Agent': 'Mozilla/5.0'}
	payload = {'voice_name': "f00002e_dnn_beta5", 'vibpow': vibpower, 'f0shift': f0shift, "alpha": synalpha}
	files = {'score': ('score.xml', xml_data.encode("utf-8"), 'application/xml')}

	# Sending post request and saving response as response object
	r = requests.post(url='https://sinsy.sp.nitech.ac.jp/api/synthesize', headers=headers, params=payload, files=files)
	html_response = r.text.split("temp/")

	# Magical scraping of the website to find the name of the wav file generated
	url_file_name = find_wav_name_on_website(html_response)

	if url_file_name is None:
		raise Exception("No wav file found on sinsy.jp :( Try again or create an issue at https://github.com/mathigatti/midi2voice/issues if the problem persists.")
	else:
		return download(url_file_name)

def find_wav_name_on_website(htmlResponse):
    data = json.loads(htmlResponse[0])
    return data["files"]["resultWav"]

def download(url_file_name):
    url = "https://sinsy.sp.nitech.ac.jp" + url_file_name
    
    with urllib.request.urlopen(url) as response:
        wav_binary = response.read()  # bytes
    
    return wav_binary


def sec_to_ticks(s: float, sec_per_beat, divisions) -> int:
    return round(s / sec_per_beat * divisions)

def parse_pitch(token: str):
    """'F#4' → ('F', 4, 1) | 'Bb3' → ('B', 3, -1) | 'D4' → ('D', 4, 0)"""
    m = re.match(r'^([A-G])([#b]?)(\d+)$', token)
    if not m:
        raise ValueError(f"Cannot parse pitch: {token!r}")
    step, acc, octave = m.group(1), m.group(2), int(m.group(3))
    alter = {'#': 1, 'b': -1, '': 0}[acc]
    return step, octave, alter

def ticks_to_notation(ticks: int):
    _TYPE_MAP = {
        64: ('breve',   False),
        48: ('whole',   True),
        32: ('whole',   False),
        24: ('half',    True),
        16: ('half',    False),
        12: ('quarter', True),
        8: ('quarter', False),
        6: ('eighth',  True),
        4: ('eighth',  False),
        3: ('16th',    True),
        2: ('16th',    False),
        1: ('32nd',    False),
    }
    if ticks in _TYPE_MAP:
        return _TYPE_MAP[ticks]
    # nearest match
    nearest = min(_TYPE_MAP, key=lambda k: abs(k - ticks))
    return _TYPE_MAP[nearest]


def syllabify(text: str):
    LANGUAGE      = 'en_US' # pyphen language for syllabification

    """Return list of (syllable_text, syllabic_position) using pyphen."""
    dic = pyphen.Pyphen(lang=LANGUAGE)
    result = []
    for word in text.split():
        parts = dic.inserted(word).split('-')
        n = len(parts)
        for i, syl in enumerate(parts):
            if n == 1:        pos = 'single'
            elif i == 0:      pos = 'begin'
            elif i < n - 1:   pos = 'middle'
            else:             pos = 'end'
            result.append((syl, pos))
    return result

def note_xml(step, octave, alter, dur_ticks,
             lyric=None, tie_start=False, tie_stop=False) -> str:
    type_str, dotted = ticks_to_notation(dur_ticks)
    lines = ['<note>']
    lines += ['  <pitch>', f'    <step>{step}</step>']
    if alter:
        lines.append(f'    <alter>{alter}</alter>')
    lines += [f'    <octave>{octave}</octave>', '  </pitch>']
    lines.append(f'  <duration>{dur_ticks}</duration>')
    if tie_stop:
        lines.append('  <tie type="stop"/>')
    if tie_start:
        lines.append('  <tie type="start"/>')
    lines.append(f'  <type>{type_str}</type>')
    if dotted:
        lines.append('  <dot/>')
    if tie_stop:
        lines.append('  <notations><tied type="stop"/></notations>')
    elif tie_start:
        lines.append('  <notations><tied type="start"/></notations>')
    if lyric:
        syl_text, syl_pos = lyric
        lines += [
            '  <lyric number="1">',
            f'    <syllabic>{syl_pos}</syllabic>',
            f'    <text>{syl_text}</text>',
            '  </lyric>',
        ]
    lines.append('</note>')
    return '\n'.join(lines)


def rest_xml(dur_ticks: int) -> str:
    type_str, dotted = ticks_to_notation(dur_ticks)
    lines = ['<note>', '  <rest/>', f'  <duration>{dur_ticks}</duration>',
             f'  <type>{type_str}</type>']
    if dotted:
        lines.append('  <dot/>')
    lines.append('</note>')
    return '\n'.join(lines)


def build_timeline(notes_data, syllables, ticks_per_meas, sec_per_beat, divisions):
    """
    Returns a flat list of segments:
      (start_tick, end_tick, pitch_tuple_or_None, lyric_or_None, is_tied_continuation)
    Rests are pitch_tuple=None. Notes crossing barlines are split into tied chunks.
    """
    # Step 1: interleave notes with rests
    raw = []   # (start_tick, end_tick, pitch, lyric)
    cursor = 0
    for i, (pitch_str, t_start, t_end) in enumerate(notes_data):
        st = sec_to_ticks(t_start, sec_per_beat, divisions)
        et = sec_to_ticks(t_end, sec_per_beat, divisions)
        if st > cursor:
            raw.append((cursor, st, None, None))   # rest
        raw.append((st, et, parse_pitch(pitch_str), syllables[i]))
        cursor = et

    # Step 2: split segments that cross barlines
    segments = []
    for t_start, t_end, pitch, lyric in raw:
        cursor = t_start
        first = True
        while cursor < t_end:
            bar_end = (cursor // ticks_per_meas + 1) * ticks_per_meas
            chunk_end = min(t_end, bar_end)
            if chunk_end > cursor:
                is_cont = not first
                segments.append((cursor, chunk_end, pitch, lyric if first else None, is_cont))
            cursor = chunk_end
            first = False
    return segments


def generate_musicxml(notes_data, lyrics_text, tempo):
    BPM           = 60_000_000 / tempo   # detected from note spacing (~0.6154 s / quarter note)
    DIVISIONS     = 8       # ticks per quarter note (must be integer)
    TIME_SIG      = 4       # beats per measure
    KEY_FIFTHS    = 1       # sharps/flats: 1 = G major / E minor (F#)

    SEC_PER_BEAT   = 60.0 / BPM           # ≈ 0.6154 s
    TICKS_PER_MEAS = DIVISIONS * TIME_SIG  # 32


    # Syllables
    syllables = syllabify(lyrics_text)

    n_notes  = len(notes_data)
    n_sylls  = len(syllables)

    if n_sylls != n_notes:
        print(f"\n⚠  Syllable / note mismatch: {n_sylls} syllables vs {n_notes} notes.")
        print("   Adjust your lyrics or use SYLLABLES_OVERRIDE.\n")
        # Pad / trim to avoid crash
        syllables = list(syllables)[:n_notes]
    
    notes_data = notes_data[:n_sylls]

    segments = build_timeline(notes_data, syllables, TICKS_PER_MEAS, SEC_PER_BEAT, DIVISIONS)

    # Group into measures
    measures = {}
    for seg in segments:
        t_start = seg[0]
        m_num   = t_start // TICKS_PER_MEAS + 1
        measures.setdefault(m_num, []).append(seg)

    # Fill any missing measure slots (gaps in measure numbers) with full rests
    if measures:
        all_m = range(min(measures), max(measures) + 1)
        for m in all_m:
            if m not in measures:
                m_start = (m - 1) * TICKS_PER_MEAS
                measures[m] = [(m_start, m_start + TICKS_PER_MEAS, None, None, False)]

    # ── Emit XML ───────────────────────────────────────────────────────────────
    out = []
    out += [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC',
        '  "-//Recordare//DTD MusicXML 3.1 Partwise//EN"',
        '  "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="3.1">',
        '  <part-list>',
        '    <score-part id="P1">',
        '      <part-name>Voice</part-name>',
        '    </score-part>',
        '  </part-list>',
        '  <part id="P1">',
    ]

    first_measure = True
    for m_num in sorted(measures.keys()):
        out.append(f'    <measure number="{m_num}">')

        if first_measure:
            out += [
                '      <attributes>',
                f'        <divisions>{DIVISIONS}</divisions>',
                '        <key>',
                f'          <fifths>{KEY_FIFTHS}</fifths>',
                '        </key>',
                '        <time>',
                f'          <beats>{TIME_SIG}</beats>',
                '          <beat-type>4</beat-type>',
                '        </time>',
                '        <clef>',
                '          <sign>G</sign>',
                '          <line>2</line>',
                '        </clef>',
                '      </attributes>',
                '      <direction placement="above">',
                '        <direction-type>',
                '          <metronome parentheses="no">',
                '            <beat-unit>quarter</beat-unit>',
                f'            <per-minute>{BPM}</per-minute>',
                '          </metronome>',
                '        </direction-type>',
                f'        <sound tempo="{BPM}"/>',
                '      </direction>',
            ]
            first_measure = False

        segs = measures[m_num]
        # Sort within measure and deduplicate potential overlap
        segs.sort(key=lambda s: s[0])

        for idx, (t_start, t_end, pitch, lyric, is_cont) in enumerate(segs):
            dur = t_end - t_start
            if dur <= 0:
                continue

            # Determine if this chunk needs a tie-start (next chunk is continuation)
            next_is_cont = (idx + 1 < len(segs) and segs[idx + 1][4])
            # Also check first segment of next measure
            next_m_segs = measures.get(m_num + 1, [])
            if not next_is_cont and next_m_segs and next_m_segs[0][4]:
                next_is_cont = True

            if pitch is None:
                elem = rest_xml(dur)
            else:
                step, octave, alter = pitch
                elem = note_xml(
                    step, octave, alter, dur,
                    lyric=lyric,
                    tie_start=next_is_cont,
                    tie_stop=is_cont,
                )

            for line in elem.split('\n'):
                out.append('      ' + line)

        out.append('    </measure>')

    out += ['  </part>', '</score-partwise>']
    return '\n'.join(out)

def convert(path_to_midi, lyrics):
    notes, tempo = convert_midi_to_notes(path_to_midi)
    xml_data = generate_musicxml(notes, lyrics, tempo)
    return renderize_voice(xml_data)

if __name__ == "__main__":
    LYRICS = """
all we have
to you i dream about is to be the same thing
to the things i do we do yeah
to you i dream about is to be the same thing
to the things i do and i do
and i'm through with that dream
and i'm through that dream
and i'll be that same
and i'll be that same girl
you and me i know
"""
    convert(r"/Users/caizhenzhi/Documents/cocolyricist/XAI-Lyricist/imagine_midi_test.mid", LYRICS)
    