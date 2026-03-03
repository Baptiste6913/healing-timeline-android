/**
 * Service Worker — Healing Timeline PWA
 * Caches shell assets for offline use and faster repeat loads.
 */

const CACHE_NAME = 'healing-timeline-v15';
const SHELL_ASSETS = [
    './',
    './index.html',
    './style.css',
    './js/app.js',
    './js/healing-model.js',
    './js/face-zones.js',
    './manifest.json',
    './icons/icon-192.png',
    './icons/icon-512.png',
];

// CDN assets (Three.js + MediaPipe) — cache on first use
const CDN_HOSTS = [
    'cdn.jsdelivr.net',
    'storage.googleapis.com',
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(SHELL_ASSETS))
            .then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(
                keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
            )
        ).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Network-first for navigation (HTML)
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request)
                .then(res => {
                    const clone = res.clone();
                    caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
                    return res;
                })
                .catch(() => caches.match('./index.html'))
        );
        return;
    }

    // Cache-first for local assets and CDN resources
    const isCDN = CDN_HOSTS.some(h => url.hostname.includes(h));
    if (url.origin === location.origin || isCDN) {
        event.respondWith(
            caches.match(event.request).then(cached => {
                if (cached) return cached;
                return fetch(event.request).then(res => {
                    if (res.ok) {
                        const clone = res.clone();
                        caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
                    }
                    return res;
                });
            })
        );
        return;
    }

    // Default: network only
    event.respondWith(fetch(event.request));
});
