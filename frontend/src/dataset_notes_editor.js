import { wireNotesEditorInteractions } from "./notes_editor_interactions.js";

export function wireDatasetNotesEditor(deps = {}) {
  const getNotesProgrammaticInput = typeof deps.getNotesProgrammaticInput === "function"
    ? deps.getNotesProgrammaticInput
    : () => false;
  const getLastSavedNotesText = typeof deps.getLastSavedNotesText === "function"
    ? deps.getLastSavedNotesText
    : () => "";
  const setNotesDirty = typeof deps.setNotesDirty === "function"
    ? deps.setNotesDirty
    : () => {};
  const updateNotesSaveUi = typeof deps.updateNotesSaveUi === "function"
    ? deps.updateNotesSaveUi
    : () => {};
  const saveNotesForCurrentContext = typeof deps.saveNotesForCurrentContext === "function"
    ? deps.saveNotesForCurrentContext
    : async () => ({ ok: false, error: "Notes save handler not configured." });
  const setStatus = typeof deps.setStatus === "function"
    ? deps.setStatus
    : () => {};

  return wireNotesEditorInteractions({
    ids: {
      inputId: "dsNotesInput",
      wrapId: "dsNotesInputWrap",
      decorId: "dsNotesDecor",
      formatToolbarId: "dsNotesFormatToolbar",
      toolbarId: "dsNotesToolbar",
      saveBtnId: "dsNotesSaveBtn",
    },
    classes: {
      tooltipClass: "dsNotesPathTooltip",
      pathTokenClass: "dsNotesPathToken",
      hoverPathClass: "isHoverPath",
    },
    getNotesProgrammaticInput,
    getLastSavedNotesText,
    setNotesDirty,
    updateNotesSaveUi,
    onSaveNotes: saveNotesForCurrentContext,
    setStatus,
    formatSaveErrorStatus: (result) => `Notes save failed: ${result?.error || "Unknown error."}`,
  });
}
