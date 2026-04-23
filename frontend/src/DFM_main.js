const params = new URL(import.meta.url).search;
const orchestratorUrl = new URL("./dfm_tabs_orchestrator.js", import.meta.url);
orchestratorUrl.search = params;

const { initDfmRatios } = await import(orchestratorUrl.toString());
initDfmRatios();
