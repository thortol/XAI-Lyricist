# XAI-Lyricist
The official repository for the paper ***[XAI-Lyricist: Improving the Singability of AI-Generated Lyrics with Prosody Explanations](https://www.ijcai.org/proceedings/2024/0872)*** by Qihao Liang, Xichu Ma, Finale Doshi-Velez, Brian Lim, and Ye Wang. This paper has been published at the *[the 33th International Joint Conference on Artificial Intelligence (IJCAI 2024), Special Track on Human-Centred Artificial Intelligence: Multidisciplinary Contours and Challenges of Next-Generation AI Research and Applications](https://ijcai24.org/call-for-papers-human-centred-artificial-intelligence/), 3rd-9th August, Jeju, South Korea.** 

## Usage

### STEP 0: Environmental Setup
```shell
export PYTHONPATH=.
```

### STEP 1: Building Dictionaries
We first construct dictionaries for both lyrics and melodies with the following code. 
```python
python ./0_build_dict/build_dictionary.py -config ./configs/configs.yaml
```

### STEP 2: Binarising Data
With the dictionaries ready, we binarise the dataset by converting lyrics and melodies to tokens.
```python
python ./1_data_binarisation/binarise.py -config ./configs/configs.yaml
```

### STEP 3: Training
```python
python ./3_train_bart/train.py -config ./configs/configs.yaml
```
### STEP 4: Inference
We provide two versions of lyrics inference, **melody-based** and **parody-based**.
#### Melody-Based Lyrics Generation
For melody-based inference, the input is a MIDI file with melody phrase boundaries marked. `imagine_midi_test.mid` provides a good example. The MIDI is first analysed and converted to a prosody template conditioning lyrics generation. The resulting lyrics are expected to share the same prosodic pattern as the melody.
```python
python ./4_infer_bart/inference.py -config ./configs/configs.yaml
```
#### Parody-Based Lyrics Generation
The parody-based inference uses lyrics as input. The system analyses the prosody of lyrics by retrieving their IPA annotation, marking each syllable with a strength and length symbol. This prosody further conditions the model to generate a new piece of lyrics with the same prosodic pattern as the input.
```
python ./4_infer_bart/inference_parody.py -config ./configs/configs.yaml
```

## API Deployment (Docker + Render)

This repository includes a FastAPI service in `api/main.py` and can be deployed to Render with Docker.

### 1. Local Docker smoke test
```bash
docker buildx build --platform linux/amd64 -t xai-lyricist-api --load .
docker run --rm -p 10000:10000 -e PORT=10000 xai-lyricist-api
```

Then check health:
```bash
curl http://localhost:10000/health
```

### 2. Option 1 (Recommended): Deploy a prebuilt image (no checkpoint in Git)

Use this path if you cannot commit `bestM2LCkpt.pt` to GitHub.

#### 2.1 Build and push image to a registry (example: Docker Hub)
Set your values:
```bash
export DOCKERHUB_USER="<dockerhub-username>"
export DOCKERHUB_PAT="<dockerhub-access-token>"
export IMAGE="${DOCKERHUB_USER}/xai-lyricist-api"
export TAG="$(date +%Y%m%d-%H%M%S)"
```

Login and push:
```bash
echo "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USER" --password-stdin
docker buildx build --platform linux/amd64 -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" --push .
```

The checkpoint is baked into the image during build (`COPY bestM2LCkpt.pt /app/bestM2LCkpt.pt`), so Render does not need it from GitHub.

#### 2.2 Create Render service from that image
1. In Render, click `New +` -> `Web Service`.
2. Choose `Deploy an existing image from a registry`.
3. Image URL: `docker.io/<dockerhub-username>/xai-lyricist-api:latest` (or a pinned tag).
4. If private, add registry credentials in Render (username + token/PAT).
5. Set:
   - Health Check Path: `/health`
   - Environment Variables:
     - `XAI_DEVICE=cpu`
     - `XAI_CHECKPOINT_PATH=/app/bestM2LCkpt.pt`
     - `XAI_DICT_PATH=/app/binary/m2l_dict.pkl`
     - `XAI_CONFIG_PATH=/app/configs/configs.yaml`
6. Deploy.

#### 2.3 Verify deployment
```bash
curl https://<your-render-service>.onrender.com/health
```

### 3. Production behavior
- Uvicorn runs on `0.0.0.0:$PORT` (Render-compatible).
- API startup is fail-fast: deployment fails if model assets cannot be loaded.

### Alternative: Git-based Render Blueprint
- `render.yaml` is included for Git-based Blueprint deployment.
- This still requires the Docker build context to include `bestM2LCkpt.pt`.
- If the checkpoint is not in the repo, prefer the prebuilt image flow above.

### Notes on Render free tier
- Free tier services can spin down when idle (cold starts).
- This model artifact is large; memory constraints may require upgrading plan for stable uptime.


### To cite this work
```
@inproceedings{ijcai2024p872,
  title     = {XAI-Lyricist: Improving the Singability of AI-Generated Lyrics with Prosody Explanations},
  author    = {Liang, Qihao and Ma, Xichu and Doshi-Velez, Finale and Lim, Brian and Wang, Ye},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {7877--7885},
  year      = {2024},
  month     = {8},
  note      = {Human-Centred AI},
  doi       = {10.24963/ijcai.2024/872},
  url       = {https://doi.org/10.24963/ijcai.2024/872},
}
```
