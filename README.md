# Meta-lingua fork for the Stanford CS cluster / miso

This fork contains some minimal changes necessary to get standard, simple configs to run on the NLP cluster and the miso queue.

For documentation of lingua, see [the original lingua repo](https://github.com/facebookresearch/lingua)

Credit for most of the changes to stool and async eval go to Yangjun.

## Quickstart

First, symlink the data dir to the shared cluster data directory
```bash
ln -sf /juice5/scr5/nlp/data/huggingface/lingua-data/ data
```
We use `uv` as our package manager. You can install dependencies with
```bash
uv sync
```

Set up the output directory. Either use a fast devices like sphinx (Be sure to change the path to your own directory! Also, `/sphinx` requires a quota which you can get from contacting CS IT.)
```bash
ln -sf /sphinx/u/thashim/lingua-checkpoints out
```

Alternatively, make an output directory on juice, but this is slow so you will need to checkpoint less often. You will want to use async checkpointing if you use this option - see notes below.
```bash
mkdir -p out/
```

Edit the config file - you should set things like the wandb project names in `apps/main/configs/llama_1B_8H200.yaml`. You can now run the sbatch example for the 1b model with 8GPUs.

Here's the example for Tatsu's setup - be sure to double check your configs -  BASE_DIR and the email should all be set to your own values. 
BASE_DIR should point to the root of your lingua-fork repo. 
```bash
export BASE_DIR=/juice5/scr5/thashim/lingua-test/lingua-fork
cd $BASE_DIR
export CONFIG_FILE=apps/main/configs/llama_1B_8H200.yaml
sbatch --mail-user=thashim@stanford.edu --export=ALL,BASE_DIR,CONFIG_FILE apps/main/configs/miso_8.slurm 
```

# Other examples

Doing learning rate sweeps is straightforward using the option override features of lingua. An example is in `scripts/lr_sweep.sh`.
```bash
scripts/lr_sweep.sh
```

Here's an example of how to run a chinchilla optimal model.
```bash
export CONFIG_FILE=apps/main/configs/llama_8H200_chinchilla.yaml
sbatch --mail-user=thashim@stanford.edu --export=ALL,BASE_DIR,CONFIG_FILE -t 1:00:00 apps/main/configs/miso_8.slurm
```

Since lingua does precise, full-state checkpointing, you can use preemptible GPUs if you want
```bash
export CONFIG_FILE=apps/main/configs/llama_1B_48G.yaml
sbatch --mail-user=thashim@stanford.edu --export=ALL,BASE_DIR,CONFIG_FILE apps/main/configs/preemptible_1.slurm
```

You can also do multi-node, 24 GPU model training for a 1B model as an example
```bash
export CONFIG_FILE=apps/main/configs/llama_1B_24H200.yaml
sbatch --mail-user=thashim@stanford.edu --export=ALL,BASE_DIR,CONFIG_FILE apps/main/configs/miso_24.slurm
```

If you want your runs to be completely deterministic, you can use the deterministic flag here. This also shows an example of using the sphinx queue instead of miso.
```bash
export CONFIG_FILE=apps/main/configs/llama_280M_48G_1.yaml
sbatch --mail-user=thashim@stanford.edu --export=ALL,BASE_DIR,CONFIG_FILE apps/main/configs/sphinx_1.slurm
```


# Other configs 
A 'fast' config that finishes in 1 hour is a 280M model which we get via
```bash
export CONFIG_FILE=apps/main/configs/llama_280M_8H200.yaml
sbatch --mail-user=thashim@stanford.edu --export=ALL,BASE_DIR,CONFIG_FILE apps/main/configs/miso_8.slurm
```

If you want to try using the preemptible queue with 4 GPUs, remember to have gradient accumulation, as you see here
```bash
export CONFIG_FILE=apps/main/configs/llama_1B_48Gx4-32acc.yaml
sbatch --mail-user=thashim@stanford.edu --export=ALL,BASE_DIR,CONFIG_FILE apps/main/configs/preemptible_4.slurm
```

Finally, miso should be able to train a 7B model as well - a chinchilla optimal model takes 9.5 days on 8 GPUs and 3+ days on 24 GPUs.
```bash
export CONFIG_FILE=apps/main/configs/llama_7B_8H200.yaml
sbatch --mail-user=thashim@stanford.edu --export=ALL,BASE_DIR,CONFIG_FILE apps/main/configs/miso_8.slurm
```

# Data related notes
Here is an example entry (from a fineweb sample) in the jsonl file. The fileloader *only* cares about fields named either `text` or `content`.
```json
{"text":"|Henry Gray (18251861). Anatomy of the Human Body. 1918.|\n|tubercle on its posterior surface, medial to the groove for the tendon of the Flexor hallucis longus. The deep fibers (anterior talotibial) are attached, above, to the tip of the medial malleolus, and, below, to the medial surface of the talus. The deltoid ligament is covered by the tendons of the Tibialis posterior and Flexor digitorum longus.|\n| The anterior and posterior talofibular and the calcaneofibular ligaments were formerly described as the three fasciculi of the external lateral ligament of the ankle-joint.|\nThe Anterior Talofibular Ligament. (ligamentum talofibulare anterius) (Fig. 355).The anterior talofibular ligament, the shortest of the three, passes from the anterior margin of the fibular malleolus, forward and medially, to the talus, in front of its lateral articular facet.\nThe Posterior Talofibular Ligament (ligamentum talofibulare posterius) (Fig. 355).The posterior talofibular ligament, the strongest and most deeply seated, runs almost horizontally from the depression at the medial and back part of the fibular malleolus to a prominent tubercle on the posterior surface of the talus immediately lateral to the groove for the tendon of the Flexor hallucis longus.\nThe Calcaneofibular Ligament (ligamentum calcaneofibulare) (Fig. 355).The calcaneofibular ligament, the longest of the three, is a narrow, rounded cord, running from the apex of the fibular malleolus downward and slightly backward to a tubercle on the lateral surface of the calcaneus. It is covered by the tendons of the Peronæi longus and brevis.\nFIG. 356 Capsule of left talocrura articulation (distended). Lateral aspect. (See enlarged image)\nSynovial Membrane (Fig. 356).The synovial membrane invests the deep surfaces of the ligaments, and sends a small process upward between the lower ends of the tibia and fibula.\nRelations.The tendons, vessels, and nerves in connection with the joint are, in front, from the medial side, the Tibialis anterior, Extensor hallucis proprius, anterior tibial vessels, deep peroneal nerve, Extensor digitorum longus, and Peronæus tertius; behind, from the medial side, the Tibialis posterior, Flexor digitorum longus, posterior tibial vessels, tibial nerve, Flexor hallucis longus; and, in the groove behind the fibular malleolus, the tendons of the Peronæi longus and brevis.\n| The arteries supplying the joint are derived from the malleolar branches of the anterior tibial and the peroneal.|\n| The nerves are derived from the deep peroneal and tibial.|\nMovements.When the body is in the erect position, the foot is at right angles to the leg. The movements of the joint are those of dorsiflexion and extension; dorsiflexion consists in the","id":"<urn:uuid:8cd958a8-013d-4fc6-91dd-ce0778145d63>","metadata":{"dump":"CC-MAIN-2018-05","url":"http://www.bartleby.com/107/pages/page351.html","file_path":"s3://commoncrawl/crawl-data/CC-MAIN-2018-05/segments/1516084891706.88/warc/CC-MAIN-20180123032443-20180123052443-00542.warc.gz","language":"en","language_score":0.7337160110473633,"token_count":721,"score":2.734375,"int_score":3}}
```

# Notes

## Disk space and fast checkpointing

Disk sizes needed for checkpoints from inspecting some runs
| Model | Size |
|-------|------|
| 7B | 85GB |
| 1B | 40GB |

This means your output directory will fill up very quickly. Async checkpointing + use of juice5 can help. You can turn this option on in the checkpointing config part as
```yaml
checkpoint:
  dump:
    every: 500
    keep: 1
  eval:
    every: 1000
    keep: 1
  async_save_mode: "shm"
  async_cleanup: true
```
The "shm" save mode works in the following way: the system will first checkpoint to `/dev/shm` which is a fast, memory-backed filesystem. Then, it will use a second thread to copy the checkpoint to the actual output directory (which is slower) and then clean up after itself when its done. `async_cleanup` handles the logic to delete old, stale checkpoints from juice - this can sometimes take a while, so it gets offloaded onto its own thread.  

Any eval jobs that rely on checkpoints will wait until the sentinel files `checkpoint_{}.done` are present, since its no longer the case that checkpoints are always present in the output directory. 

Use of async checkpointing can have its own complications - the threads may sometimes not properly cleanup, and leave weird files in `/dev/shm` or crash due to lack of space on `shm`. Setting `async_save_mode: null` will disable this.

## Downloading and shuffling data

To download and shuffle a dataset on a small, fast disk (such as `/scr-ssd` on one of the Miso machines) and later transfer to a slower, larger disk (such as `/juice5b`), you can use the `setup/download_prepare_hf_data.py` script.

```bash
uv run python setup/download_prepare_hf_data.py dclm_baseline_1.0_10prct 999 --tmp_dir=/scr-ssd/nlp/tmp --data_dir=/scr-ssd/nlp/data --seed=42 --nchunks=8 --final_data_dir=/juice5b/scr5b/nlp/data/huggingface/lingua-data --clear_work_dir_after_transfer_to_final
```

This will
1. First download the data to `/scr-ssd/nlp/data/dclm_baseline_1.0_10prct` and shuffle it, writing the shuffled data to `/scr-ssd/nlp/data/dclm_baseline_1.0_10prct_shuffled`.
2. Copy the shuffled data to the slow disk at `/juice5b/scr5b/nlp/data/huggingface/lingua-data/dclm_baseline_1.0_10prct_shuffled`.
3. Clear the work directories at `/scr-ssd/nlp/data/dclm_baseline_1.0_10prct` and `/scr-ssd/nlp/data/dclm_baseline_1.0_10prct_shuffled` if the `--clear_work_dir_after_transfer_to_final` flag is set.

# Changelog

12/17/2024
- Added async checkpointing
- Some chinchilla scaling recipes

12/10/2024 (v0)
- Added sbatch scripts and setup that works without stool
- Traps SIGTERM in addition to SIGUSR2 since the stanford SLURM config does not send user signals on preemption
- Modifications to the async eval script to work with the cluster, and also to fix bugs in `stool.py`, `eval.py` and `distributed.py`
  - Fixed checkpointing to wait for eval to complete - this means disk usage may be higher than the `keep` parameter.
- Added a deterministic mode, and verified deterministic training when the flag is on. Also verified that the model can be loaded from a checkpoint and continue exactly