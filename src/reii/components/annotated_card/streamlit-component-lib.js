/**
 * Minimal self-contained Streamlit component API shim.
 *
 * The original index.html loaded this library from a CDN URL
 * (https://cdn.jsdelivr.net/npm/streamlit-component-lib/dist/streamlit-component-lib.js)
 * that never existed in the npm package (the package ships ES modules:
 * dist/streamlit.js, dist/ArrowTable.js, ...). As a result the `Streamlit`
 * object was always undefined and the component never signalled ready.
 *
 * This file implements the exact subset of the official API used by this
 * component, with the same postMessage protocol:
 *
 *   -> streamlit:componentReady  { apiVersion: 1 }
 *   -> streamlit:setFrameHeight  { height }
 *   -> streamlit:setComponentValue { value, dataType: "json" }
 *   <- streamlit:render          { args, disabled, theme }
 *
 * No external dependencies; works offline (Docker, proxies, etc.).
 */
(function () {
  "use strict";

  var RENDER_EVENT = "streamlit:render";
  var events = new EventTarget();
  var registeredMessageListener = false;
  var lastFrameHeight = null;

  function sendBackMsg(type, data) {
    var msg = { isStreamlitMessage: true, type: type };
    for (var k in data) {
      if (Object.prototype.hasOwnProperty.call(data, k)) {
        msg[k] = data[k];
      }
    }
    window.parent.postMessage(msg, "*");
  }

  function setComponentReady() {
    if (!registeredMessageListener) {
      window.addEventListener("message", onMessageEvent);
      registeredMessageListener = true;
    }
    sendBackMsg("streamlit:componentReady", { apiVersion: 1 });
  }

  function setFrameHeight(height) {
    if (height === undefined) {
      height = document.body.scrollHeight;
    }
    if (height === lastFrameHeight) {
      return;
    }
    lastFrameHeight = height;
    sendBackMsg("streamlit:setFrameHeight", { height: height });
  }

  function setComponentValue(value) {
    sendBackMsg("streamlit:setComponentValue", {
      value: value,
      dataType: "json",
    });
  }

  function onMessageEvent(event) {
    var data = event.data || {};
    if (data["type"] === RENDER_EVENT) {
      onRenderMessage(data);
    }
  }

  function onRenderMessage(data) {
    var args = data["args"] || {};
    var disabled = Boolean(data["disabled"]);
    var theme = data["theme"];
    if (theme) {
      injectTheme(theme);
    }
    var event = new CustomEvent(RENDER_EVENT, {
      detail: { disabled: disabled, args: args, theme: theme },
    });
    events.dispatchEvent(event);
  }

  function injectTheme(theme) {
    var style = document.createElement("style");
    document.head.appendChild(style);
    style.innerHTML =
      ":root{" +
      "--primary-color:" + theme.primaryColor + ";" +
      "--background-color:" + theme.backgroundColor + ";" +
      "--secondary-background-color:" + theme.secondaryBackgroundColor + ";" +
      "--text-color:" + theme.textColor + ";" +
      "--font:" + theme.font + ";}" +
      "body{background-color:var(--background-color);color:var(--text-color);}";
  }

  window.Streamlit = {
    API_VERSION: 1,
    RENDER_EVENT: RENDER_EVENT,
    events: events,
    setComponentReady: setComponentReady,
    setFrameHeight: setFrameHeight,
    setComponentValue: setComponentValue,
  };
})();
