# Data and resource sources

## Reused original micro-CT

| Domain | Original repository identifier | Local preparation catalog |
|---|---|---|
| Main data | [10.6084/m9.figshare.c.4190738](https://doi.org/10.6084/m9.figshare.c.4190738) | `code/main_raw_preparation.json` |
| Bentheimer | [10.17612/KT0B-SZ28](https://doi.org/10.17612/KT0B-SZ28) | `code/external_training.json` |
| WF1 | [10.17612/P7HT11](https://doi.org/10.17612/P7HT11) | `code/external_training.json` |

Original images are not included here. Follow each provider's current access
and licence conditions. Public access is not a blanket licence for mirroring
the images or every derived artifact. Preserve the providers' attribution.
The preparation catalogs specify the expected files and pixels; the entry
points do not guess correspondence by similar filenames.

### Provider licences and file entries

Checked on 2026-10-07:

- **Main data:** the [DataCite record](https://api.datacite.org/dois/10.6084/m9.figshare.c.4190738)
  and the [data author's institutional record](https://research.tees.ac.uk/en/datasets/x-ray-computed-micro-tomography-dataset-for-nanoparticle-based-oi-2/)
  identify CC BY 4.0. The Figshare landing page and API returned HTTP 403 during
  this check; no successful file download is claimed.
- **Bentheimer:** the [provider's project page](https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-229)
  explicitly displays ODC-BY 1.0, although the DataCite rights field is empty.
  The [Core 1 high-resolution tomographic data entry](https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-229/digital_dataset/62b98ecf-19b4-4945-b150-47d5c3141b10)
  lists `Core_1_imbFo0_3_45_zoom.raw` (file-list page 2).
- **WF1:** the [provider's project page](https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-167)
  and DataCite identify ODC-BY 1.0. The [WF1 middle entry](https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-167/digital_dataset/395e9005-3dfe-4b89-bda6-14562605d6b2)
  lists `WF1_middle_1030x1033x1000_16bu.raw`: 1030 wide by 1033 high,
  1000 slices, unsigned 16-bit little-endian, 5-micrometre voxel spacing.

The external file entries are public without sign-in. Their metadata and
filenames were checked, not the multi-gigabyte download contents. Keep
acquisition metadata tied to the specific volume: the Bentheimer provider
warns that generic image-header voxel sizes may differ from the scan
resolution encoded in the filename. This check does not alter the project's
fixed preprocessing or calibration.

The [ODC-BY text](https://opendatacommons.org/licenses/by/1-0/) distinguishes
database rights from rights in individual contents. Credit Jackson, Lin and
Krevor for Bentheimer, and Scanziani, Singh and Blunt for WF1, with their dataset
DOIs and the ODC-BY notice when applicable. A new licence for author-created
annotations does not replace these upstream notices or relicense CT pixels.

## This study's resources

- **Annotations:** 7,130 eight-class masks with sample pairing, split and prior
  metadata. The external reference set was produced using an image-assisted
  workflow and accepted after domain-expert review before adaptation. The
  detailed construction is described in the manuscript and
  [data dictionary](code/DATA_DICTIONARY.md). It is not the providers' original
  phase-label taxonomy.
- **Weights:** selected three-stage model states for the three main seeds,
  six original external configurations and three configurations in the complete
  WF1 repetition. The exact case/stage map is in the model catalogs.
- **Figure source values:** per-seed and per-class numerical inputs and
  plotting scripts under `figures/`. Figure IDs and units are defined in
  [SOURCE_DATA.md](figures/SOURCE_DATA.md).
- **Image-derived figure assets:** fixed crops, archived labels/predictions,
  activations and a validation volume. These remain separate local-only
  reproduction inputs pending redistribution confirmation; they are not
  contained in the code/numerical bundle.

The label and weight attachment identities are in `release_assets.json`.
Versioned resources: <https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0>. Author-created resources use
CC BY 4.0, with scope and original-data notices in
[RESOURCE_LICENSE.md](RESOURCE_LICENSE.md); author-written software uses MIT.
No model input, reference mask or experimental result changes during packaging.

