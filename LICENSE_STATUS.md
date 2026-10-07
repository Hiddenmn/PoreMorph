# Licence status

The authors confirmed the following licence scheme on 2026-10-07 for
PoreMorph v0.1.0: <https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0>.

| Material | Licence and notice |
|---|---|
| Author-written software and software documentation | [MIT](LICENSE) |
| Author-generated annotations, splits, priors, metadata and numerical source data | [CC BY 4.0](LICENSE-DATA.txt), with [scope and attribution](RESOURCE_LICENSE.md) |
| Author-trained selected model states | [CC BY 4.0](LICENSE-DATA.txt), with [scope and attribution](RESOURCE_LICENSE.md) |
| Unchanged Nature Skills alignment helper | Apache 2.0, as specified below |
| Original CT data, original phase labels and other third-party materials | Their own provider terms; not relicensed here |

Numerical inputs in `figures/source_data/` and the author-generated scientific records
in `code/` follow the resource licence, not the default software licence.
Implementation/configuration files and software documentation follow MIT.
The author grant applies only to rights held by the PoreMorph contributors;
existing upstream notices remain in effect.

## Legacy project material

The pre-deletion MM-PoreFormer repository at commit
`6b513f3e03c4356a85dfc8f60dcb24def693104a` supplied an MIT software licence with
the notice `Copyright (c) 2026 Qisong Zhou`. Its exact text is retained in
[LICENSE-MM-PoreFormer.txt](LICENSE-MM-PoreFormer.txt) for reused legacy
software portions. The old repository's `DATA_AVAILABILITY.md` separately
identified its released reviewed masks as CC BY 4.0, with attribution to the
project and the original Pak et al. dataset.

The new author licence choice is stated above separately from these historical
terms. A software licence is not automatically a licence for every Release
attachment. Image-derived assets are excluded from this release.

## Third-party helper

One third-party file has a verified existing licence:

- `figures/scripts/audit_panel_alignment.py` is byte-identical to the file in
  Yuan1z0825/nature-skills at commit
  `7d5f160ebfe8033375b4afef7c5911aa4203c983`.
- That repository revision supplies Apache License 2.0. Its exact licence
  text is included in `LICENSES/NatureSkills-Apache-2.0.txt`.
- Source: <https://github.com/Yuan1z0825/nature-skills/blob/7d5f160ebfe8033375b4afef7c5911aa4203c983/skills/nature-figure/scripts/audit_panel_alignment.py>.
- Licence: <https://github.com/Yuan1z0825/nature-skills/blob/7d5f160ebfe8033375b4afef7c5911aa4203c983/LICENSE>.

The upstream file is not modified. This notice and the included licence apply
to that helper, not automatically to the rest of PoreMorph or any CT-derived
material. Third-party runtime libraries retain their own licences and are not
vendored here. The original CT providers' conditions remain separate.

## Original datasets

The Pak et al. collection is registered as CC BY 4.0. The live Bentheimer and
WF1 repository pages both display ODC-BY 1.0, as checked on 2026-10-07.
The Bentheimer DataCite record omits the rights field; the live provider page
supplies it. Source links and the relevant raw-file entries are listed in
[DATA_SOURCES.md](DATA_SOURCES.md).

ODC-BY governs database rights and does not itself grant all rights in each
individual image; see its Sections 2.4 and 4.3. Preserve the provider's
attribution and applicable terms when distributing derived resources.
The new release will not mirror the original CT volumes. Image-derived figure
inputs remain local pending their applicable redistribution conditions.

