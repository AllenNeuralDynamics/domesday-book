"""Synthetic test corpus for evaluating domesday retrieval.

Mimics the kind of tacit knowledge a neuroscience research team accumulates:
dataset caveats, pipeline gotchas, experimental conclusions, parameter choices,
and institutional knowledge that doesn't live in any paper.

Each snippet has an ID prefix (t01–t30) for stable reference in eval queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TestSnippet:
    id_prefix: str
    text: str
    author: str
    project: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """A query with known-relevant snippets and optionally a reference answer."""

    query: str
    relevant_ids: list[str]  # id_prefixes that SHOULD be retrieved
    irrelevant_ids: list[str] = field(default_factory=list)  # should NOT appear
    category: str = "general"  # retrieval | negative | synthesis | specificity
    reference_answer: str | None = None


# ---------------------------------------------------------------------------
# Test snippets: 30 entries across several topic clusters
# ---------------------------------------------------------------------------

SNIPPETS: list[TestSnippet] = [
    # --- Cluster: VBO dataset issues ---
    TestSnippet(
        id_prefix="t01",
        text=(
            "The VBO dataset has an off-by-one error in frame timestamps before "
            "2023-06-01. All analyses using timing data from sessions before this "
            "date need to subtract one frame (16.67ms at 60Hz) from stimulus onset "
            "times. This was fixed in VBO v2.3."
        ),
        author="sarah",
        project="vbo",
    ),
    TestSnippet(
        id_prefix="t02",
        text=(
            "VBO sessions 847-892 have corrupted eye-tracking data. The IR LED "
            "array failed intermittently, causing pupil tracking dropouts of 2-5 "
            "seconds. These sessions should be excluded from any analysis that "
            "depends on gaze position or pupil diameter."
        ),
        author="marcus",
        project="vbo",
        tags=["vbo", "eye-tracking", "data-quality"],
    ),
    TestSnippet(
        id_prefix="t03",
        text=(
            "VBO running speed signals above 80 cm/s are unreliable. The rotary "
            "encoder saturates at high speeds and the signal wraps around. We clip "
            "running speed to 80 cm/s in the standard preprocessing pipeline."
        ),
        author="sarah",
        project="vbo",
        tags=["vbo", "running", "preprocessing"],
    ),
    TestSnippet(
        id_prefix="t04",
        text=(
            "When joining VBO stimulus tables with neural data, use the "
            "'stimulus_presentation_id' column, NOT 'stimulus_id'. The latter "
            "is a template ID shared across sessions while the former is unique "
            "per presentation. Confusing these was the source of the Wang et al. "
            "replication failure last quarter."
        ),
        author="ben",
        project="vbo",
        tags=["vbo", "data-wrangling", "gotcha"],
    ),
    # --- Cluster: Two-photon imaging ---
    TestSnippet(
        id_prefix="t05",
        text=(
            "Our Scientifica two-photon rigs show a ~3 pixel drift in X over the "
            "first 20 minutes of each session as the galvo mirrors warm up. Always "
            "discard the first 20 minutes or apply motion correction before "
            "extracting ROIs. Rig 2 is worse than Rig 1."
        ),
        author="jun",
        project="vbo",
        tags=["two-photon", "motion", "hardware"],
    ),
    TestSnippet(
        id_prefix="t06",
        text=(
            "Suite2p cell detection works well for L2/3 neurons but over-segments "
            "L5 apical dendrites into multiple ROIs. For deep imaging sessions, "
            "manually merge ROIs that track the same dendrite — check the "
            "correlation of their dF/F traces (>0.85 usually means same cell)."
        ),
        author="jun",
        project="vbo",
        tags=["two-photon", "suite2p", "segmentation"],
    ),
    TestSnippet(
        id_prefix="t07",
        text=(
            "The neuropil correction coefficient (r) in suite2p defaults to 0.7 "
            "but we've found 0.5 works better for our GCaMP8m mice. Higher values "
            "over-subtract and create artifactual negative transients. We validated "
            "this against simultaneous electrophysiology in 3 animals."
        ),
        author="priya",
        project="vbo",
        tags=["two-photon", "suite2p", "neuropil", "gcamp"],
    ),
    # --- Cluster: Behavioral training ---
    TestSnippet(
        id_prefix="t08",
        text=(
            "Mice on the visual discrimination task plateau at ~75% accuracy if "
            "trained with the standard 4-second ITI. Reducing ITI to 2 seconds "
            "improved asymptotic performance to ~85% in 8/10 animals. The shorter "
            "ITI seems to maintain engagement better."
        ),
        author="marcus",
        project="vbo",
        tags=["behavior", "training", "visual-discrimination"],
    ),
    TestSnippet(
        id_prefix="t09",
        text=(
            "Water restriction to 85% body weight is our standard, but animals "
            "under 20g at baseline should use 90% instead. We lost two animals "
            "last year that were restricted too aggressively at low body weight. "
            "Always check the welfare chart before adjusting restriction level."
        ),
        author="sarah",
        project="vbo",
        tags=["behavior", "welfare", "water-restriction"],
    ),
    TestSnippet(
        id_prefix="t10",
        text=(
            "The auditory go/no-go task has a strong side bias for the first "
            "50-100 trials of each session. This resolves on its own but inflates "
            "early-session false alarm rates. Best practice: exclude the first 80 "
            "trials from all auditory analyses."
        ),
        author="marcus",
        project="vbo",
        tags=["behavior", "auditory", "bias"],
    ),
    # --- Cluster: Electrophysiology ---
    TestSnippet(
        id_prefix="t11",
        text=(
            "Neuropixels 2.0 probes from lot #NP2-2024-03 have elevated noise "
            "on bank 1 (channels 384-767). Affected probes: serial numbers "
            "SN20240301 through SN20240315. Use the alternate bank configuration "
            "or apply the Steinmetz denoising pipeline."
        ),
        author="jun",
        project="vbo",
        tags=["ephys", "neuropixels", "hardware", "noise"],
    ),
    TestSnippet(
        id_prefix="t12",
        text=(
            "Kilosort4 consistently over-splits fast-spiking interneurons into "
            "2-3 units. After sorting, always check clusters with ISI violations "
            ">1% — these are usually split FSIs that should be merged. The "
            "auto_merge parameter in KS4 helps but doesn't fully solve it."
        ),
        author="priya",
        project="vbo",
        tags=["ephys", "kilosort", "spike-sorting"],
    ),
    TestSnippet(
        id_prefix="t13",
        text=(
            "Our chronic Neuropixels implants show ~15% channel attrition per "
            "month. By month 3, you're typically down to 300-350 usable channels "
            "from the original 384. Plan recording schedules accordingly — the "
            "most important experiments should run first."
        ),
        author="jun",
        project="vbo",
        tags=["ephys", "neuropixels", "chronic", "planning"],
    ),
    # --- Cluster: Analysis pipeline ---
    TestSnippet(
        id_prefix="t14",
        text=(
            "The standard z-scoring for dF/F traces uses the first 60 seconds "
            "as baseline. This is a problem for sessions where the mouse is "
            "already running at session start — use the median of the full "
            "trace instead, or identify quiescent periods automatically."
        ),
        author="priya",
        project="vbo",
        tags=["analysis", "normalization", "calcium"],
    ),
    TestSnippet(
        id_prefix="t15",
        text=(
            "Our GPU cluster (nodes gpu01-gpu08) has a CUDA version mismatch: "
            "nodes 01-04 run CUDA 12.1, nodes 05-08 run CUDA 11.8. PyTorch "
            "models trained on one set won't load on the other without "
            "recompilation. Always specify the node range in SLURM with "
            "--nodelist or use the conda env that matches."
        ),
        author="ben",
        project="vbo",
        tags=["infrastructure", "gpu", "cuda", "slurm"],
    ),
    TestSnippet(
        id_prefix="t16",
        text=(
            "Pandas groupby().apply() silently drops groups that return empty "
            "DataFrames. This burned us in the orientation tuning analysis — "
            "unresponsive neurons were dropped entirely instead of getting NaN "
            "tuning curves. Use groupby().apply() with include_groups=False "
            "or switch to a loop."
        ),
        author="ben",
        project="vbo",
        tags=["analysis", "pandas", "gotcha"],
    ),
    # --- Cluster: Specific experimental findings ---
    TestSnippet(
        id_prefix="t17",
        text=(
            "Conclusion from the Dec 2024 analysis: V1 neurons show reliable "
            "mismatch responses only when the visual flow deviation exceeds ~15 "
            "degrees from predicted. Below that threshold, responses are "
            "indistinguishable from noise. This constrains our stimulus design "
            "for the predictive coding experiments."
        ),
        author="priya",
        project="vbo",
        tags=["v1", "mismatch", "predictive-coding", "finding"],
    ),
    TestSnippet(
        id_prefix="t18",
        text=(
            "SST interneurons in L2/3 of V1 show paradoxical excitation during "
            "locomotion onset — opposite to what Bharioke et al. 2022 reported. "
            "We've replicated this in 6 animals. Possible explanation: our mice "
            "are head-fixed on a floating ball vs. their linear treadmill. The "
            "vestibular input is different."
        ),
        author="jun",
        project="vbo",
        tags=["v1", "interneurons", "sst", "locomotion", "finding"],
    ),
    TestSnippet(
        id_prefix="t19",
        text=(
            "LGN recordings: we see clear orientation tuning in ~30% of relay "
            "cells, much higher than the textbook 5-10%. Likely because our "
            "stimuli are high spatial frequency (0.08 cpd) which preferentially "
            "drives orientation-tuned LGN cells. Important to note when comparing "
            "to literature that typically uses 0.02-0.04 cpd."
        ),
        author="priya",
        project="vbo",
        tags=["lgn", "orientation", "finding", "stimuli"],
    ),
    # --- Cluster: Data management ---
    TestSnippet(
        id_prefix="t20",
        text=(
            "LIMS session IDs and NWB file names use different conventions. "
            "LIMS: numeric (e.g. 1094870112). NWB: includes date and mouse "
            "(e.g. sub-620333_ses-1094870112). The mapping table is at "
            "//allen/programs/braintv/workgroups/nc-ophys/lims_nwb_map.csv "
            "— do NOT try to parse NWB filenames to extract session IDs."
        ),
        author="sarah",
        project="vbo",
        tags=["data-management", "lims", "nwb", "gotcha"],
    ),
    TestSnippet(
        id_prefix="t21",
        text=(
            "All NWB files from before 2024-01 use the old coordinate system "
            "where anterior is negative. Files after that date use the Allen CCF "
            "convention where anterior is positive. Check the 'reference_frame' "
            "field in the NWB metadata — if it says 'custom', it's the old system."
        ),
        author="sarah",
        project="vbo",
        tags=["data-management", "nwb", "coordinates", "bug"],
    ),
    # --- Cluster: Software / tools ---
    TestSnippet(
        id_prefix="t22",
        text=(
            "allensdk version 2.16.0 has a bug in BrainObservatoryCache that "
            "returns duplicate rows when filtering by container_id. Fixed in "
            "2.16.1. Pin your version or deduplicate results with "
            "df.drop_duplicates(subset='cell_specimen_id')."
        ),
        author="ben",
        project="vbo",
        tags=["allensdk", "bug", "software"],
    ),
    TestSnippet(
        id_prefix="t23",
        text=(
            "For video analysis of mouse behavior, DeepLabCut works fine for "
            "body parts but fails badly on whisker tracking. Use JARVIS or "
            "WhiskerMan for whisker kinematics instead. DLC's skeleton model "
            "can't handle the whisker's thin, high-curvature geometry."
        ),
        author="marcus",
        project="vbo",
        tags=["software", "deeplabcut", "whiskers", "video"],
    ),
    # --- Cluster: Histology / anatomy ---
    TestSnippet(
        id_prefix="t24",
        text=(
            "Our standard perfusion protocol with 4% PFA gives inconsistent "
            "clearing for iDISCO+. Switching to 3% PFA + 0.1% glutaraldehyde "
            "dramatically improved tissue clearing while preserving GFP "
            "fluorescence. Protocol updated in the shared lab notebook 2024-09."
        ),
        author="jun",
        project="vbo",
        tags=["histology", "clearing", "idisco", "protocol"],
    ),
    TestSnippet(
        id_prefix="t25",
        text=(
            "The Allen CCF v3 atlas has a known misalignment in posterior VISp "
            "(primary visual cortex) near the VISp/VISpm border. Registrations "
            "in this region can be off by up to 200μm. For critical analyses, "
            "manually verify probe tracks against cytoarchitecture."
        ),
        author="priya",
        project="vbo",
        tags=["anatomy", "ccf", "atlas", "registration"],
    ),
    # --- Cluster: Stimulus / display ---
    TestSnippet(
        id_prefix="t26",
        text=(
            "Monitor gamma correction LUT was recalibrated on 2024-11-15. "
            "Sessions before this date have slightly incorrect luminance values "
            "(measured ~8% deviation at mid-gray). Probably not enough to "
            "affect most analyses but worth noting for contrast sensitivity work."
        ),
        author="marcus",
        project="vbo",
        tags=["stimulus", "display", "calibration"],
    ),
    TestSnippet(
        id_prefix="t27",
        text=(
            "The ASUS VG248QE monitors we use have a 1-frame input lag that's "
            "NOT accounted for in the photodiode timestamp correction. True "
            "stimulus onset is ~16.7ms later than what the stimulus log says. "
            "This stacks with the VBO off-by-one for pre-2023-06 data."
        ),
        author="ben",
        project="vbo",
        tags=["stimulus", "display", "timing", "latency"],
    ),
    # --- Cluster: Statistical methods ---
    TestSnippet(
        id_prefix="t28",
        text=(
            "For trial-averaged neural responses, use the cluster-based "
            "permutation test (Maris & Oostenveld 2007) rather than pointwise "
            "t-tests. The latter gives massive false positive rates with our "
            "autocorrelated calcium data. We implemented this in "
            "analysis_utils.cluster_permutation_test()."
        ),
        author="priya",
        project="vbo",
        tags=["statistics", "methods", "calcium", "analysis"],
    ),
    TestSnippet(
        id_prefix="t29",
        text=(
            "When fitting tuning curves, the von Mises function fails for "
            "neurons with bimodal orientation tuning (common in L4). Use a "
            "mixture of two von Mises instead, or just report the circular "
            "variance which doesn't assume unimodality."
        ),
        author="priya",
        project="vbo",
        tags=["statistics", "tuning-curves", "orientation"],
    ),
    # --- Standalone / misc ---
    TestSnippet(
        id_prefix="t30",
        text=(
            "Lab meeting consensus 2025-01: we're standardizing on Python 3.12 "
            "and numpy 2.x for all new analysis code. Existing pipelines stay "
            "on numpy 1.x until explicitly migrated. Don't mix environments — "
            "pickle files are not compatible across numpy major versions."
        ),
        author="sarah",
        project="vbo",
        tags=["infrastructure", "python", "standards"],
    ),
]


# ---------------------------------------------------------------------------
# Evaluation queries with ground truth
# ---------------------------------------------------------------------------

EVAL_QUERIES: list[EvalQuery] = [
    # --- Precision: specific factual retrieval ---
    EvalQuery(
        query="What's the timestamp bug in the VBO dataset?",
        relevant_ids=["t01"],
        category="retrieval",
        reference_answer=(
            "There's an off-by-one error in frame timestamps before 2023-06-01. "
            "Subtract one frame (16.67ms at 60Hz) from stimulus onset times."
        ),
    ),
    EvalQuery(
        query="Which VBO sessions have bad eye tracking?",
        relevant_ids=["t02"],
        category="retrieval",
    ),
    EvalQuery(
        query="What's the right join key for VBO stimulus tables?",
        relevant_ids=["t04"],
        category="retrieval",
    ),
    EvalQuery(
        query="How long should I wait before taking two-photon data?",
        relevant_ids=["t05"],
        category="retrieval",
        reference_answer="20 minutes, due to galvo mirror warm-up drift.",
    ),
    EvalQuery(
        query="neuropil correction coefficient for GCaMP8m",
        relevant_ids=["t07"],
        category="retrieval",
    ),
    EvalQuery(
        query="Why did the Wang replication fail?",
        relevant_ids=["t04"],
        category="retrieval",
    ),
    # --- Synthesis: queries that should pull multiple snippets ---
    EvalQuery(
        query="What are all the known timing issues I should worry about?",
        relevant_ids=["t01", "t27"],
        category="synthesis",
        reference_answer=(
            "Two known timing issues: the VBO off-by-one (pre-2023-06), and the "
            "monitor input lag of ~16.7ms not in photodiode correction. They stack."
        ),
    ),
    EvalQuery(
        query="What are the problems with the VBO dataset?",
        relevant_ids=["t01", "t02", "t03", "t04"],
        category="synthesis",
    ),
    EvalQuery(
        query="What should I know about suite2p settings for our data?",
        relevant_ids=["t06", "t07"],
        category="synthesis",
    ),
    EvalQuery(
        query="What are our experimental findings about V1?",
        relevant_ids=["t17", "t18"],
        category="synthesis",
    ),
    EvalQuery(
        query="What infrastructure gotchas should new lab members know about?",
        relevant_ids=["t15", "t20", "t21", "t30"],
        category="synthesis",
    ),
    # --- Specificity: should retrieve one thing, not similar-but-wrong ---
    EvalQuery(
        query="Kilosort splitting fast-spiking neurons",
        relevant_ids=["t12"],
        irrelevant_ids=["t06"],  # suite2p over-segmentation is different
        category="specificity",
    ),
    EvalQuery(
        query="How does running speed affect data quality?",
        relevant_ids=["t03"],
        irrelevant_ids=["t18"],  # locomotion onset is different topic
        category="specificity",
    ),
    EvalQuery(
        query="NWB coordinate system issue",
        relevant_ids=["t21"],
        irrelevant_ids=["t25"],  # CCF atlas misalignment is related but different
        category="specificity",
    ),
    # --- Negative: no relevant content should be returned ---
    EvalQuery(
        query="What's the best restaurant near the Allen Institute?",
        relevant_ids=[],
        category="negative",
    ),
    EvalQuery(
        query="How do I set up a Docker container for our pipeline?",
        relevant_ids=[],
        category="negative",
    ),
    EvalQuery(
        query="What's the latest on the Mars rover mission?",
        relevant_ids=[],
        category="negative",
    ),
    EvalQuery(
        query="optimal learning rate for ResNet-50 training",
        relevant_ids=[],
        category="negative",
    ),
    # --- Cross-domain: queries touching multiple clusters ---
    EvalQuery(
        query="What display and monitor issues affect timing?",
        relevant_ids=["t26", "t27"],
        category="synthesis",
    ),
    EvalQuery(
        query="What bugs have we found in third-party software?",
        relevant_ids=["t22", "t12"],  # allensdk bug, kilosort issue
        category="synthesis",
    ),
    EvalQuery(
        query="What should I know before running a chronic recording experiment?",
        relevant_ids=["t13", "t11"],
        category="synthesis",
    ),
]