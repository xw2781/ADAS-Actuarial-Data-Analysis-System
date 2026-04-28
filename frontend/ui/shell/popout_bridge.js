/**
 * Pop-Out Bridge — BroadcastChannel wrapper for pop-out window communication.
 *
 * Channel name: "arcrho:popout:{inst}" (unique per tab instance)
 *
 * Message types:
 *   popout-ready       — pop-out loaded and ready
 *   popout-closed      — pop-out closing, restore tab in main shell
 *   relay-to-shell     — iframe→shell message relayed from pop-out
 *   relay-to-iframe    — shell→iframe message relayed to pop-out
 */

const CHANNEL_PREFIX = "arcrho:popout:";

/**
 * Creates a BroadcastChannel for a specific tab instance.
 * @param {string} inst — tab instance ID (e.g. "ds_1", "dfm_2", "wf_3")
 * @returns {{ send(msg: object): void, onMessage(cb: (msg: object) => void): void, close(): void }}
 */
export function createPopoutChannel(inst) {
  const channel = new BroadcastChannel(`${CHANNEL_PREFIX}${inst}`);
  let handler = null;

  channel.onmessage = (e) => {
    if (handler && e.data) handler(e.data);
  };

  return {
    send(msg) {
      try { channel.postMessage(msg); } catch { /* closed */ }
    },
    onMessage(cb) {
      handler = cb;
    },
    close() {
      handler = null;
      try { channel.close(); } catch { /* already closed */ }
    },
  };
}
