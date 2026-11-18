const storageKey = "exoplanet-explorer.preferences.v1";
let previousView;
let previousPreview;
const catalogueScroll = { Explore: 0, Saved: 0 };

export default function ({ data, setStateValue, setTriggerValue }) {
    if (!data.initialized) {
        let preferences = {};
        try {
            preferences = JSON.parse(localStorage.getItem(storageKey)) || {};
        } catch {
            // Private browsing or disabled storage should still allow exploration.
        }
        setStateValue("preferences", preferences);
    } else {
        try {
            localStorage.setItem(storageKey, JSON.stringify(data.preferences));
        } catch {
            // The session continues if storage is unavailable.
        }
    }

    const scrollArea = document.querySelector('[data-testid="stMain"]');
    const isCatalogue = Object.hasOwn(catalogueScroll, data.view);
    const trackScroll = () => {
        if (isCatalogue) catalogueScroll[data.view] = scrollArea.scrollTop;
    };
    let resizeTimer;
    if (scrollArea) {
        scrollArea.addEventListener("scroll", trackScroll, { passive: true });
        if (previousView === "Profile" && isCatalogue) {
            const position = catalogueScroll[data.view];
            resizeTimer = setTimeout(() => scrollArea.scrollTo({ top: position }), 250);
        } else if (data.preview && data.preview !== previousPreview && innerWidth <= 640) {
            resizeTimer = setTimeout(() => {
                document.querySelector(".st-key-catalogue-preview")?.scrollIntoView({ block: "start" });
            }, 300);
        }
    }
    previousView = data.view;
    previousPreview = data.preview;

    const onKey = (event) => {
        if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey ||
            event.target instanceof Element &&
            event.target.closest("input, textarea, select, [contenteditable=true], [role=dialog]")) return;
        if (document.querySelector("[role=dialog]")) return;
        if (event.key === "/") {
            const search = document.querySelector('.st-key-home-toolbar input, input[aria-label="Find a planet"]');
            if (search) {
                event.preventDefault();
                search.focus();
            }
        }
        if (event.key.toLowerCase() === "f" && !event.shiftKey) {
            const filters = document.querySelector('.st-key-open_filters [data-testid^="stBaseButton"]');
            if (filters) {
                event.preventDefault();
                filters.click();
            }
        }
    };
    const onHistory = (event) => {
        // The app owns query-string routes. Pass the address bar directly so
        // a pending widget rerun cannot overwrite a Back or Forward action.
        event.stopImmediatePropagation();
        setTriggerValue("route", location.search);
    };
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("popstate", onHistory, true);
    return () => {
        document.removeEventListener("keydown", onKey, true);
        window.removeEventListener("popstate", onHistory, true);
        scrollArea?.removeEventListener("scroll", trackScroll);
        clearTimeout(resizeTimer);
    };
}
