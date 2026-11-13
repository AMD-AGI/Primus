# ShareGPT smoke fixture provenance

Source: [Aeala/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered),
revision `8b0048ad6ae8c22f46a78c15559dec98feef5539`, file
`ShareGPT_V4.3_unfiltered_cleaned_split.json`.
The dataset card identifies this as a reupload further cleaned by gozfarb and
lists Apache-2.0. The declared license is retained in `LICENSE-ShareGPT.txt`.
This license notice applies to the dataset fixture, not Primus source code.

`sharegpt_800.jsonl` contains 800 conversations selected from the prior eligible
smoke-test pool. Selection excluded conversations matching common email, phone,
credential, private-key, and IP-address patterns. This is a targeted screen,
not a guarantee that arbitrary conversation text contains no personal information.
The conversation text is unchanged. The only transformation is the structural
conversion from `from`/`value` to `role`/`content`, with human mapped to user and
gpt mapped to assistant. Empty/malformed/nonalternating conversations were excluded.

Each JSONL row includes its source `id`; the file itself defines the row order.
The source revision, source SHA-256, and output SHA-256 are recorded in
`../../../scripts/specforge_smoke_fixture.json`. Use the checked-in fixture
directly; no preparation script or separate ID list is needed.

The fixture is 4,243,587 bytes. It is stored as plain JSONL for review, with LF
line endings enforced by `.gitattributes`. The large-file pre-commit exception
is restricted to this one checksum-pinned fixture.
