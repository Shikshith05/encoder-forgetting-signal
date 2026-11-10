# A Pre-Fine-Tuning Signal for Encoder-Level Forgetting Asymmetry in Vision-Language Continual Learning

PES University Capstone PW26_VK_06.

| Part | Owner |
|---|---|
| A Signal | Kishan Kabadi S |
| B Fine-tuning | Shreyas K M |
| C Swap + Shapley | Shikshith V |
| D Confidence (ECE) | Mythri S |

Shreyas's Part B is ready in [the self-contained Kaggle/Colab notebook](notebooks/B_finetune_Shreyas.ipynb).
On Kaggle, import it, select a GPU, enable Internet, and use **Save Version → Save & Run All**.
Follow the [Kaggle steps through result verification and GitHub handoff](docs/KAGGLE_PART_B.md).
On Colab, select a GPU and run its cells; Drive is optional. The notebook
embeds all required Part B code and saves the four
named checkpoints, measured training times, all four test accuracies and reproducible
data manifests to `OUTPUT_DIR`. Download the artifact archive before ending a local
Colab runtime. Kaggle writes `part_b_artifacts.zip` under `/kaggle/working` for the
saved version's Output tab. See [the implementation notes](docs/SHREYAS_PART_B.md).

For a local CUDA environment:

```bash
pip install -r requirements.txt
python scripts/run_finetune.py --out-dir checkpoints/quickgelu --fallback
```

The optional fallback adds a matched EuroSAT pair at lr 3e-5 only if base Pets
forgetting is below 1 percentage point. Base results are retained. No GPU results
are included until the real experiments are run.

After changing shared source, rebuild the embedded notebook with
`python scripts/build_finetune_notebook.py`. Validate with
`python -m pytest -q`; Part B tests run on CPU without downloading datasets/weights.
