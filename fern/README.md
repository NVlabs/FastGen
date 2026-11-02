# FastGen documentation

Edit component READMEs for generated guides, or `docs/*.mdx` for authored pages.
Add new README sources to [scripts/build_docs.py](scripts/build_docs.py) and
pages to [docs.yml](docs.yml). Generated files in `docs/generated/`
are ignored by Git; rebuild them after editing a source README.

## Validate and preview

Requires Python 3.10+ and Node.js 22+. No FastGen installation or GPU is needed.
Run from the checkout root:

```bash
npm install -g pnpm@10.20.0
npm install -g "fern-api@$(node -p 'require("./fern/fern.config.json").version')"
python3 fern/scripts/build_docs.py
python3 -B tests/test_fern.py
python3 fern/scripts/check-docs-paths fern
fern check
fern docs md check
fern docs dev
```

Open the URL printed by `fern docs dev`. Local previews work without signing in
and may use Fern's default theme. Pull requests run the documentation checks
automatically.
