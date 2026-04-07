"""Patch Mamba3 Triton kernels for fp16 (Turing) compatibility.

Patches bf16 -> fp16 for SSM_States and step kernel.
Sporadic NaN on long sequences is handled by GradScaler.
"""

import os
import sys

site = sys.argv[1]  # mamba_ssm package directory

# Patches: (filename, find, replace)
patches = []

# bf16 -> fp16 for tensors that Turing can't do in bf16
patches.append(("ops/triton/mamba3/mamba3_siso_fwd.py", "torch.bfloat16", "torch.float16"))
patches.append(("ops/triton/mamba3/mamba3_siso_step.py", "tl.bfloat16", "tl.float16"))
patches.append(("ops/triton/mamba3/mamba3_mimo_rotary_step.py", "torch.bfloat16", "torch.float16"))

# Apply patches
modified = set()
for relpath, find, replace in patches:
    path = os.path.join(site, relpath)
    with open(path) as f:
        src = f.read()
    count = src.count(find)
    if count == 0:
        print(f"WARNING: '{find}' not found in {relpath}")
        continue
    src = src.replace(find, replace)
    with open(path, "w") as f:
        f.write(src)
    modified.add(relpath)
    print(f"  {relpath}: replaced {count}x: {find[:50]}...")

print(f"Patched {len(modified)} files")
