# av-SALMONN
av-SALMONN: Speech-Enhanced Audio-Visual Large Language Models

<a href='https://651c4fdabc0fbe05d6.gradio.live'><img src='https://img.shields.io/badge/gradio-demo-blue'></a>

Button Specifications:

`Clear All`: clear chat history as well as all modality inputs. **Please always use clear all before you want to upload or update any image, audio or video** 

`Clear history`: only clear chat history. The modality input will remain unchanged unless you click `Clear All`.

`Submit`: submit the text in the text box to get a response

`Resubmit`: clear the previous conversation turn and then submit the text in the text box

`maximum length`, `top p` and `temperature` have their own individual meanings

Examples mentioned in the paper are provided. Please feel free to start with those.


We provide the script for evaluating speech (LibriSpeech) and audio (AudioCaps) as single-modal tasks using Video-LLaMA. Please find codes in `infer_batch.sh` and `video_llama/`
We provide the generated results for LibriSpeech (`librispeech.json` and `librispeech_finetuned.json` for finetuning 50k steps on LibriSpeech) and AudioCaps (`audiocaps.json`)
<a href='https://651c4fdabc0fbe05d6.gradio.live'><img src='https://img.shields.io/badge/gradio-demo-blue'></a>
