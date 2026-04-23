/*
===============================================================================
DFM Notes Tab - textarea get/set/wire
===============================================================================
*/
import { markDfmDirty } from "./dfm_state.js";
import { wireNotesEditorInteractions } from "./notes_editor_interactions.js";

let dfmNotesProgrammaticInput = false;
let dfmLastSavedNotesText = "";

function setStatus(text) {
  try {
    window.parent.postMessage({ type: "adas:status", text: String(text || "") }, "*");
  } catch {
    // ignore
  }
}

export function getDfmNotesText() {
  return document.getElementById("dfmNotesInput")?.value ?? "";
}

export function setDfmNotesText(value) {
  const notesInput = document.getElementById("dfmNotesInput");
  const nextText = typeof value === "string" ? value : "";
  dfmLastSavedNotesText = nextText;
  if (!notesInput) return;
  dfmNotesProgrammaticInput = true;
  notesInput.value = nextText;
  notesInput.dispatchEvent(new Event("input", { bubbles: true }));
  dfmNotesProgrammaticInput = false;
}

export function wireNotesInput() {
  wireNotesEditorInteractions({
    ids: {
      inputId: "dfmNotesInput",
      wrapId: "dfmNotesInputWrap",
      decorId: "dfmNotesDecor",
      formatToolbarId: "dfmNotesFormatToolbar",
      // DFM keeps its own save flow, so no save button/toolbar ids are provided.
    },
    classes: {
      tooltipClass: "dfmNotesPathTooltip",
      pathTokenClass: "dfmNotesPathToken",
      hoverPathClass: "isHoverPath",
    },
    getNotesProgrammaticInput: () => dfmNotesProgrammaticInput,
    getLastSavedNotesText: () => dfmLastSavedNotesText,
    setNotesDirty: () => {
      markDfmDirty();
    },
    updateNotesSaveUi: () => {},
    onSaveNotes: async () => ({ ok: true }),
    setStatus,
    formatSaveErrorStatus: (result) => `DFM notes save failed: ${result?.error || "Unknown error."}`,
  });
}
