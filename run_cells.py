# Execute selected code cells of the notebook in one namespace (used to run the long model-fitting cell in resumable chunks).
import nbformat, sys, time
idx = [int(x) for x in sys.argv[1].split(",")]
cs = [c.source for c in nbformat.read("Paper3_OnionRetraining.ipynb", as_version=4).cells if c.cell_type == "code"]
ns = {"__name__": "__main__", "display": print}
for i in idx:
    t = time.time(); exec(cs[i], ns); print(f"[cell {i}] {time.time()-t:.1f}s", flush=True)
