# LEAF-1 Checkpoint

This directory contains the pretrained LEAF-1 checkpoint used for cfDNA layer-4
manuscript runs. It is a HuggingFace checkpoint directory and should be passed to
inference commands as:

```bash
leaf1-extract --bert_checkpoint checkpoint ...
```

Checkpoint payload files are tracked with Git LFS. After cloning, fetch the LFS
payload if Git did not download it automatically:

```bash
git lfs install
git lfs pull --include="checkpoint/**"
```

Files required for frozen-encoder inference:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

Trainer state files such as `optimizer.pt`, `scheduler.pt`, `trainer_state.json`,
and `rng_state_*.pth` are included for audit and resume, but are not required for
embedding extraction. `training_args.bin` is omitted from the release checkpoint
because it only stores training-launch metadata.

`model.safetensors` and `optimizer.pt` are larger than GitHub's regular file
limit, so they must stay under Git LFS and should not be committed as ordinary
git blobs.
