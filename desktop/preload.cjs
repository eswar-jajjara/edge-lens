'use strict';
const { contextBridge, ipcRenderer } = require('electron');

// Expose one fixed provider action, never arbitrary URLs or Node.js primitives.
contextBridge.exposeInMainWorld('EdgeLensDesktop', Object.freeze({
  signInEdgeImpulse: () => ipcRenderer.invoke('edgelens:edge-impulse-sign-in'),
}));
