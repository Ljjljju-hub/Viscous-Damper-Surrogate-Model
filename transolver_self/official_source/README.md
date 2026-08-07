# Official source checkout

The upstream THUML repository is checked out locally at `Transolver/` and is
ignored by the parent repository so its nested `.git` directory is not
accidentally committed.

- Repository: https://github.com/thuml/Transolver
- Verified revision: `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`
- License: MIT

Refresh the checkout with:

```powershell
git -C transolver_self\official_source\Transolver pull --ff-only
```

The runnable project does not import from this checkout. The adapted,
dependency-light implementation is kept under `transolver_self/model/` so a
normal clone of the damper project remains complete.
