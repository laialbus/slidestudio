# Schema-level field length constants.
#
# These values define the maximum character counts for text fields across
# all Pydantic schemas. They are data contract constraints — they control
# what the pipeline accepts as valid LLM output and trigger the retry loop
# when exceeded. They are not user-facing tuning knobs; do not move them
# to config.py.
#
# Change a value here when the semantic definition of that field changes
# (e.g. "summary" evolves from a sentence to a paragraph), not to work
# around a specific model's verbosity.

THESIS_MAX                  = 400    # single sentence capturing the core argument
SECTION_SUMMARY_MAX         = 1000   # ~3-4 sentences describing a document section
SLIDE_BODY_MAX              = 1200   # ~4 sentences of self-contained prose explanation
DECK_TITLE_MAX              = 60     # short deck title shown in the viewer header
PLANNED_SLIDE_ANNOTATION_MAX = 200   # per-slide planning notes (intention, emphasis)
DECK_FEEDBACK_MAX           = 300    # overall deck-level feedback from the Critic
MAX_FIGURES_PER_SLIDE       = 3      # most figures a single slide may request/carry (stacked)
