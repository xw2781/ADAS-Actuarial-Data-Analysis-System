export function publishDfmInputHelpers(deps) {
  const {
    getResolvedProjectValue,
    getResolvedReservingClassValue,
    getDisplayProjectValue,
    getDisplayReservingClassValue,
    getDisplayTriValue,
    isInputDefaultBound,
  } = deps;

  try {
    window.ADA_GET_DFM_INPUTS = () => ({
      resolved: {
        project: getResolvedProjectValue(),
        reservingClass: getResolvedReservingClassValue(),
        tri: getDisplayTriValue(),
      },
      display: {
        project: getDisplayProjectValue(),
        reservingClass: getDisplayReservingClassValue(),
        tri: getDisplayTriValue(),
      },
      defaults: {
        projectDefault: isInputDefaultBound(document.getElementById("projectSelect")),
        reservingClassDefault: isInputDefaultBound(document.getElementById("pathInput")),
      },
    });
  } catch {
    // ignore
  }
}

export function wireDatasetHostBridge(deps) {
  const { getTriInputsForStorage, instanceId, redrawChartSafely } = deps;

  window.addEventListener("message", (e) => {
    if (e?.data?.type === "adas:get-dataset-settings") {
      const settings = getTriInputsForStorage();
      window.parent.postMessage(
        {
          type: "adas:dataset-settings",
          requestId: e.data.requestId,
          stepId: instanceId,
          settings,
        },
        "*"
      );
      return;
    }

    if (e?.data?.type === "adas:tab-activated") {
      // Only redraw when THIS tab becomes active
      requestAnimationFrame(() => {
        requestAnimationFrame(redrawChartSafely);
      });
    }
  });
}
