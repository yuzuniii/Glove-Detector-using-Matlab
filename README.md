# Glove Defect Detection System (GDD)
### CT036-3-IPPR — Image Processing and Pattern Recognition
### Asia Pacific University of Technology & Innovation (APU)

---

## Overview

The Glove Defect Detection System (GDD) is a fully automatic image processing pipeline built in MATLAB that detects surface and structural defects on industrial gloves. The system identifies the glove type from a raw image, segments the glove from the background, and routes it through a type-specific defect detection workstream — all without any manual configuration.

The system supports four glove types and thirteen defect categories across four workstreams.

---

## Supported Glove Types and Defects

| Workstream | Glove Type | Defects Detected |
|---|---|---|
| WS1 | White Latex | Stain, Dirty, Spotting |
| WS2 | Black Nitrile | Stain, Tear, Incomplete Beading |
| WS3 | Blue Nitrile | Stain, Scratch, Fold Damage, Inside-Out |
| WS4 | Cotton Work | Stain, Burn Hole, Uneven Weave |

---

## Project Structure

```
GDD_Project/
│
├── main.m                          Entry point — run this to launch the GUI
│
├── gui/
│   └── GloveDefectGUI.m            Professional dark-theme detection GUI
│
├── defects/
│   ├── color/
│   │   ├── detectStain.m           Shared stain detector (all glove types)
│   │   ├── detectBurnHole.m        WS4 — burn hole via background void + dark ring
│   │   └── detectColourContamination.m  WS4 — hue deviation detection
│   ├── geometry/
│   │   ├── detectTear.m            WS2 — bright hole inside black glove mask
│   │   └── detectIncompleteBeading.m   WS2 — ragged cuff boundary analysis
│   ├── shape/
│   │   └── detectInsideOut.m       WS3 — saturation bimodality detection
│   └── texture/
│       ├── detectScratch.m         WS3 — directional linear morphological filter
│       ├── detectDamagedByFold.m   WS3 — Hough line crease detection
│       └── detectUnevenWeave.m     WS4 — local texture variance anomaly
│
├── segmentation/
│   ├── segmentGlove.m              Router — calls type-specific segmenter
│   ├── segmentWhiteGlove.m         LAB Mahalanobis distance segmentation
│   ├── segmentBlackGlove.m         HSV brightness threshold segmentation
│   ├── segmentBlueGlove.m          Hue-based segmentation
│   └── segmentCottonGlove.m        LAB distance from background segmentation
│
├── gloveClassification/
│   ├── detectGloveType.m           HSV + texture variance type classifier
│   ├── classifyColorDefect.m       WS1 stain/dirty/spotting scorer
│   └── classifyTextureDefect.m     WS3 texture classifier
│
├── features/
│   └── extractTextureFeatures.m    Texture feature extractor for WS3
│
├── utils/
│   ├── analyzeGlove.m              Master orchestrator for the full pipeline
│   ├── analyzeWS1.m                WS1 white latex pipeline
│   ├── analyzeWS2.m                WS2 black nitrile pipeline
│   ├── analyzeWS3.m                WS3 blue nitrile pipeline
│   └── analyzeWS4.m                WS4 cotton work glove pipeline
│
└── dataset/
    ├── testing/                    Test images for each workstream
    └── tuning/                     Images used during parameter tuning
```

---

## How to Run

1. Open MATLAB and navigate to the `GDD_Project` root folder
2. Run `main.m`
3. The GUI will launch automatically
4. Click **Browse Image...** to load a glove image
5. Click **ANALYSE** to run the full detection pipeline
6. Results appear in the right panel with bounding box annotation

---

## Pipeline Architecture

```
main.m
  └── GloveDefectGUI
        └── analyzeGlove (orchestrator)
              ├── detectGloveType        (Step 1 — type identification)
              ├── segmentGlove           (Step 2 — glove isolation)
              ├── analyzeWS1/2/3/4       (Step 3 — workstream routing)
              │     ├── WS1: classifyColorDefect → detectStain
              │     ├── WS2: detectTear, detectIncompleteBeading
              │     ├── WS3: detectScratch, detectDamagedByFold, detectInsideOut
              │     └── WS4: detectBurnHole, detectUnevenWeave
              └── detectStain            (Step 4 — shared stain detection)
```

---

## Detection Methods Summary

### Glove Type Detection
Analyses the centre 60% of the raw image in HSV colour space. Uses hue percentage, brightness, saturation, and texture variance to score each glove type. Texture variance is the key discriminator between white latex (smooth, low variance) and cotton knit (yarn grid, high variance).

### Segmentation
Each glove type uses a dedicated segmenter. White latex uses LAB Mahalanobis distance from corner-sampled background. Cotton uses LAB distance with edge-connected background removal. Blue and black use hue and brightness thresholding respectively.

### WS1 — White Latex
Extracts ten colour and texture features then applies a rule-based scorer to classify Stain (dominant dark blob), Dirty (globally dark surface), or Spotting (many small scattered dots). The shared `detectStain` runs only when the classifier says stain.

### WS2 — Black Nitrile
`detectTear` finds bright interior voids whose LAB colour matches the background (background visible through the hole). `detectIncompleteBeading` auto-detects the cuff edge orientation then measures boundary roughness, notch depth, and gap fraction.

### WS3 — Blue Nitrile
`detectScratch` uses directional morphological opening (blackhat + line SE at 8 angles) to isolate continuous linear dark marks from the grip texture dots, then uses Hough with large FillGap to bridge fragments into one dominant line. `detectDamagedByFold` uses Hough line transform to find long horizontal crease lines spanning the palm. `detectInsideOut` uses saturation bimodality — an inside-out glove shows its white inner surface creating two distinct saturation populations.

### WS4 — Cotton Work Glove
`detectBurnHole` finds background-coloured voids inside the glove mask, applies morphological closing, then confirms with circularity and dark ring checks (charred fabric surrounds the hole). `detectUnevenWeave` computes local texture variance using stdfilt, finds the palm region baseline, and detects patches significantly above baseline indicating coarse open weave.

---

## Key Design Decisions

**Shared stain detector** — `detectStain` runs across all workstreams using LAB colour space baseline deviation. It is skipped when a higher-priority defect is already found to avoid double-firing.

**Stain skip gates** — each workstream informs the orchestrator when a high-confidence defect is found, preventing `detectStain` from overriding the correct result.

**No hardcoded pixel thresholds** — all size thresholds (min area, disk radius, line length) scale proportionally with the glove bounding box or image dimensions.

**Texture variance for type detection** — the key breakthrough for separating white latex from cream cotton was measuring `stdfilt` variance on the centre crop. Smooth rubber scores below 0.050; knit yarn weave scores above 0.055.

---

## Requirements

- MATLAB R2020b or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox (for `graythresh`, `regionprops`)

---

## Test Images

| Image | Glove Type | Expected Result |
|---|---|---|
| glovestain.png | White Latex | Stain |
| dirtygloves.png | White Latex | Dirty |
| glovespotting.png | White Latex | Spotting |
| glovetear.png | Black Nitrile | Tear |
| incompleteglove.png | Black Nitrile | Incomplete Beading |
| blackstain.jpg | Black Nitrile | Stain |
| bluescratch.png | Blue Nitrile | Scratch |
| damagebyfold.png | Blue Nitrile | Fold Damage |
| insideout.png | Blue Nitrile | Inside-Out |
| bluestain.png | Blue Nitrile | Stain |
| cottonstain.png | Cotton Work | Stain |
| cottonburnhole.png | Cotton Work | Burn Hole |
| unevenweave.png | Cotton Work | Uneven Weave |

---

## Authors

Developed for CT036-3-IPPR — Image Processing and Pattern Recognition  
Asia Pacific University of Technology & Innovation (APU)
