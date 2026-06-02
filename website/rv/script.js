let table = document.getElementById("csv-container");
let input_container = document.getElementById("search-holder");
let input_box = document.getElementById("input-box");
let last_updated = document.getElementById("last-updated");
let page_status = document.getElementById("page-status");
let lastUpdatedDate = null;
const headers_array = ["RA", "DEC", "PARALLAX", "PMRA", "PMDEC", "PERIOD", "ECCENTRICITY", "M1", "M2"];
const MEDIA_HEADERS = new Set(["RV PLOT", "RV FIT", "FLUX PLOT", "SOURCE IMAGE"]);
let colByHeader = {};
const ASSET_CACHE_BUST = String(Date.now());
const RECENT_APF_DAYS = 30;
const RECENT_KECK_DAYS = 30;
const RECENT_WEEK_DAYS = 7;
const INCLUDE_KECK_ONLY_ROWS = false;
const KECK_GAIA_IDS = new Set();
const KECK_TARGETS = new Map();
const SIMBAD_GAIA_IDS = new Set();
const COMPLETE_ORBIT_DR4_HEADER = "WILL HAVE COMPLETE ORBIT BY DR4";
const COMPLETE_ORBIT_DR4_LEGACY_HEADER = "WILL HAVE COMPLETE ORBIT BY NOVEMBER";
const YES_NO_PARAMS = [
    { header: "BROAD LINES (HOT STAR/RAPID ROTATION)", label: "Broad lines", toggleId: "toggle-param-1", countId: "count-param-1", datasetKey: "param1", urlKey: "p1" },
    { header: "COMPLETE ORBIT/MINIMA TO MAXIMA", label: "Complete orbit now", toggleId: "toggle-param-2", countId: "count-param-2", datasetKey: "param2", urlKey: "p2" },
    { header: "JERK STAR", label: "Jerk star", toggleId: "toggle-param-3", countId: "count-param-3", datasetKey: "param3", urlKey: "p3" },
    { header: "SHOWS CLEAR VARIABILITY", label: "Clear variability", toggleId: "toggle-param-4", countId: "count-param-4", datasetKey: "param4", urlKey: "p4" },
    { header: COMPLETE_ORBIT_DR4_HEADER, legacyHeaders: [COMPLETE_ORBIT_DR4_LEGACY_HEADER], label: "Complete orbit by DR4", toggleId: "toggle-param-5", countId: "count-param-5", datasetKey: "param5", urlKey: "p5" },
    { header: "LOW S/N", label: "Low S/N", toggleId: "toggle-param-6", countId: "count-param-6", datasetKey: "param6", urlKey: "p6" }
];

let sortColumn = null;
let sortDirection = "asc";
let pageSize = "all";
let currentPage = 1;
let suppressUrlSync = false;
let lastMatchedRows = [];
let apfCountSortDirection = "desc";
let keckCountSortDirection = "desc";
let apfRecentSortDirection = "asc";
let keckRecentSortDirection = "asc";

const headerDisplayMap = {
    "GAIA NAME": "Gaia DR3",
    "RA (deg)": "RA (deg)",
    "DEC (deg)": "Dec (deg)",
    "PARALLAX (mas)": "Parallax (mas)",
    "PMRA (mas/yr)": "PMRA (mas/yr)",
    "PMDEC (mas/yr)": "PMDec (mas/yr)",
    "PERIOD (days)": "Period (days)",
    "ECCENTRICITY": "Eccentricity",
    "M1 (Msun)": "Luminous M1<br>(M<sub>⊙</sub>)",
    "M2 (Msun)": "Dark M2<br>(M<sub>⊙</sub>)",
    "M2sin i (Msun)": "M2 sin i<br>(M<sub>⊙</sub>)",
    "(M2sin i)/(sin i) (Msun)": "M2 at i<br>(M<sub>⊙</sub>)",
    "RV PLOT": "RV Curve",
    "RV FIT": "RV Fit",
    "FLUX PLOT": "H-beta",
    "SOURCE IMAGE": "UV Image",
    "DATA PRODUCTS": "Data Products",
    "BROAD LINES (HOT STAR/RAPID ROTATION)": "Broad lines",
    "COMPLETE ORBIT/MINIMA TO MAXIMA": "Complete orbit now",
    "JERK STAR": "Jerk star",
    "SHOWS CLEAR VARIABILITY": "Clear variability",
    "WILL HAVE COMPLETE ORBIT BY NOVEMBER": "Complete orbit by DR4",
    "WILL HAVE COMPLETE ORBIT BY DR4": "Complete orbit by DR4",
    "LOW S/N": "Low S/N"
};

function parseUrlState() {
    const params = new URLSearchParams(window.location.search);
    const state = {
        q: "",
        rows: "all",
        page: 1,
        sort: null,
        dir: "asc",
        rv: false,
        flux: false,
        source: false,
        apf: false,
        keck: false,
        rapf: false,
        rkeck: false,
        w7apf: false,
        w7keck: false,
        ranges: {
            RA: ["", ""],
            DEC: ["", ""],
            PARALLAX: ["", ""],
            PMRA: ["", ""],
            PMDEC: ["", ""],
            PERIOD: ["", ""],
            ECCENTRICITY: ["", ""],
            M1: ["", ""],
            M2: ["", ""]
        }
    };
    for (let i = 0; i < YES_NO_PARAMS.length; i++) {
        state[YES_NO_PARAMS[i].datasetKey] = "";
    }

    state.q = params.get("q") || "";
    state.rows = params.get("rows") || "all";
    const parsedPage = parseInt(params.get("page") || "1", 10);
    state.page = Number.isNaN(parsedPage) || parsedPage < 1 ? 1 : parsedPage;
    state.sort = params.get("sort");
    state.dir = params.get("dir") === "desc" ? "desc" : "asc";
    state.rv = params.get("rv") === "1";
    state.flux = params.get("flux") === "1";
    state.source = params.get("source") === "1";
    state.apf = params.get("apf") === "1";
    state.keck = params.get("keck") === "1";
    state.rapf = params.get("rapf") === "1";
    state.rkeck = params.get("rkeck") === "1";
    state.w7apf = params.get("w7apf") === "1";
    state.w7keck = params.get("w7keck") === "1";

    const rangeKeys = [
        ["RA", "ra"], ["DEC", "dec"], ["PARALLAX", "parallax"],
        ["PMRA", "pmra"], ["PMDEC", "pmdec"], ["PERIOD", "period"],
        ["ECCENTRICITY", "ecc"], ["M1", "m1"], ["M2", "m2"]
    ];
    for (let i = 0; i < rangeKeys.length; i++) {
        const label = rangeKeys[i][0];
        const short = rangeKeys[i][1];
        state.ranges[label] = [
            params.get(`${short}_min`) || "",
            params.get(`${short}_max`) || ""
        ];
    }

    for (let i = 0; i < YES_NO_PARAMS.length; i++) {
        const raw = String(params.get(YES_NO_PARAMS[i].urlKey) || "").trim().toUpperCase();
        state[YES_NO_PARAMS[i].datasetKey] = (raw === "Y" || raw === "N") ? raw : "";
    }

    return state;
}

function normalizeYesNoValue(value) {
    const v = String(value ?? "").trim().toUpperCase();
    if (v === "Y" || v === "YES" || v === "1" || v === "TRUE" || v === "T") {
        return "Y";
    }
    return "N";
}

function getParamFilterValue(param) {
    const control = document.getElementById(param.toggleId);
    if (!control) {
        return "";
    }
    const raw = String(control.value ?? "").trim().toUpperCase();
    if (raw === "Y" || raw === "N") {
        return raw;
    }
    return "";
}

function ensureYesNoColumns(tableData) {
    if (!Array.isArray(tableData) || tableData.length === 0 || !Array.isArray(tableData[0])) {
        return tableData;
    }

    const headerRow = tableData[0];
    for (let i = 0; i < YES_NO_PARAMS.length; i++) {
        const param = YES_NO_PARAMS[i];
        if (!headerRow.includes(param.header)) {
            let legacyIdx = -1;
            const legacyHeaders = Array.isArray(param.legacyHeaders) ? param.legacyHeaders : [];
            for (let j = 0; j < legacyHeaders.length; j++) {
                const idx = headerRow.indexOf(legacyHeaders[j]);
                if (idx >= 0) {
                    legacyIdx = idx;
                    break;
                }
            }

            if (legacyIdx >= 0) {
                // Rename legacy column header in-place to avoid duplicate columns.
                headerRow[legacyIdx] = param.header;
            } else {
                headerRow.push(param.header);
                const newIdx = headerRow.length - 1;
                for (let rowIdx = 1; rowIdx < tableData.length; rowIdx++) {
                    const row = tableData[rowIdx];
                    if (!Array.isArray(row)) {
                        continue;
                    }
                    while (row.length <= newIdx) {
                        row.push("N");
                    }
                }
            }
        }
    }

    for (let rowIdx = 1; rowIdx < tableData.length; rowIdx++) {
        const row = tableData[rowIdx];
        if (!Array.isArray(row)) {
            continue;
        }
        const isEmpty = row.length === 0 || row.every(cell => String(cell ?? "").trim() === "");
        if (isEmpty) {
            continue;
        }
        while (row.length < headerRow.length) {
            row.push("N");
        }
        for (let i = 0; i < YES_NO_PARAMS.length; i++) {
            const colIdx = headerRow.indexOf(YES_NO_PARAMS[i].header);
            row[colIdx] = normalizeYesNoValue(row[colIdx]);
        }
    }

    return tableData;
}

const initialState = parseUrlState();
if (window.location.search) {
    history.replaceState(null, "", window.location.pathname);
}

function formatDateTimeUTC(d) {
    const year = d.getUTCFullYear();
    const month = String(d.getUTCMonth() + 1).padStart(2, "0");
    const day = String(d.getUTCDate()).padStart(2, "0");
    const hours = String(d.getUTCHours()).padStart(2, "0");
    const mins = String(d.getUTCMinutes()).padStart(2, "0");
    const secs = String(d.getUTCSeconds()).padStart(2, "0");
    return `${year}-${month}-${day} ${hours}:${mins}:${secs}`;
}

function renderLastUpdated() {
    if (!last_updated) {
        return;
    }
    if (!lastUpdatedDate) {
        last_updated.innerHTML = "Last updated: unavailable";
        return;
    }
    const ageHours = Math.max(0, (Date.now() - lastUpdatedDate.getTime()) / 3600000);
    let freshnessClass = "fresh";
    let freshnessLabel = "Up to date";
    if (ageHours >= 72) {
        freshnessClass = "old";
        freshnessLabel = "Old";
    } else if (ageHours >= 24) {
        freshnessClass = "stale";
        freshnessLabel = "Stale";
    }
    last_updated.innerHTML = `Last updated: ${formatDateTimeUTC(lastUpdatedDate)} UTC (${ageHours.toFixed(2)} hr ago) <span id=\"freshness-badge\" class=\"${freshnessClass}\">${freshnessLabel}</span>`;
}

function considerLastUpdated(lastModifiedHeader) {
    if (!lastModifiedHeader) {
        return;
    }
    const d = new Date(lastModifiedHeader);
    if (Number.isNaN(d.getTime())) {
        return;
    }
    if (!lastUpdatedDate || d.getTime() > lastUpdatedDate.getTime()) {
        lastUpdatedDate = d;
    }
}

async function refreshLastUpdatedFromWebsiteFiles() {
    const files = [
        "index.html",
        "script.js",
        "style.css",
        "tables/data.csv",
        "tables/keck_targets.csv"
    ];
    const tasks = files.map(file =>
        fetch(file, { method: "HEAD", cache: "no-store" })
            .then(resp => {
                if (resp.ok) {
                    considerLastUpdated(resp.headers.get("last-modified"));
                }
            })
            .catch(() => {})
    );
    await Promise.allSettled(tasks);
    if (!lastUpdatedDate) {
        lastUpdatedDate = new Date();
    }
    renderLastUpdated();
}

function buildMediaCellHtml(text) {
    if (typeof text !== "string") {
        return text;
    }

    const trimmed = text.trim();
    if (!trimmed || trimmed.toUpperCase() === "N/A") {
        return text;
    }

    const srcMatch = trimmed.match(/src=['"]([^'"]+)['"]/i);
    if (srcMatch && srcMatch[1]) {
        return `<a href="${srcMatch[1]}" target="_blank" rel="noopener noreferrer">${trimmed}</a>`;
    }

    if (/^https?:\/\/\S+$/i.test(trimmed) && /\.(png|jpg|jpeg|gif|webp)(\?.*)?$/i.test(trimmed)) {
        return `<a href="${trimmed}" target="_blank" rel="noopener noreferrer"><img src="${trimmed}" alt="plot"></a>`;
    }

    return text;
}

function hasNonNAContent(value) {
    if (typeof value !== "string") {
        return false;
    }
    const trimmed = value.trim();
    return trimmed !== "" && trimmed.toUpperCase() !== "N/A";
}

function buildDataProductsHtml(dataProductsCell, fluxCell, uvImageCell) {
    if (typeof dataProductsCell !== "string") {
        return dataProductsCell;
    }

    const hrefMatch = dataProductsCell.match(/href=['"]([^'"]+)['"]/i);
    if (!hrefMatch || !hrefMatch[1]) {
        return dataProductsCell;
    }

    const base = hrefMatch[1].replace(/\/+$/, "");
    const apfUrl = `${base}/Gaia/`;
    const swiftUrl = `${base}/Swift/`;
    const hasSwift = hasNonNAContent(fluxCell) || hasNonNAContent(uvImageCell);

    let html = `<span class="product-links"><a href="${apfUrl}" target="_blank" rel="noopener noreferrer">APF</a>`;
    if (hasSwift) {
        html += ` <a href="${swiftUrl}" target="_blank" rel="noopener noreferrer">Swift</a>`;
    }
    html += `</span>`;
    return html;
}

function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

function buildGaiaNameCellHtml(text) {
    const raw = String(text || "");
    const cleaned = raw.replace(/^Gaia\s+DR3\s+/i, "").trim();
    const idMatch = cleaned.match(/(\d{8,})/);
    const copyValue = idMatch ? idMatch[1] : cleaned;
    if (!idMatch || !copyValue) {
        return cleaned;
    }
    const safeText = escapeHtml(cleaned);
    const safeCopy = escapeHtml(copyValue);
    let display = `<span class=\"gaia-id-text\">${safeText}</span>`;
    if (SIMBAD_GAIA_IDS.has(copyValue)) {
        const simbadUrl = `https://simbad.u-strasbg.fr/simbad/sim-id?Ident=${encodeURIComponent(`Gaia DR3 ${copyValue}`)}`;
        display = `<a class=\"gaia-id-text\" href=\"${simbadUrl}\" target=\"_blank\" rel=\"noopener noreferrer\">${safeText}</a>`;
    }
    return `<div class=\"gaia-id-wrap\">${display}<button class=\"copy-gaia-btn\" data-copy=\"${safeCopy}\" title=\"Copy Gaia ID\">Copy</button></div>`;
}

async function loadOptionalSimbadCatalog() {
    const path = "tables/simbad_gaia_ids.csv";
    try {
        const resp = await fetch(path, { cache: "no-store" });
        if (!resp.ok) {
            return;
        }
        const txt = await resp.text();
        const lines = txt.split(/\r?\n/);
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line || line.startsWith("#")) {
                continue;
            }
            const id = line.split(",")[0].trim().replace(/^['"]|['"]$/g, "");
            if (/^\d{8,}$/.test(id)) {
                SIMBAD_GAIA_IDS.add(id);
            }
        }
    } catch {
        // Optional file; no SIMBAD links if unavailable.
    }
}

function rebuildColumnIndex(headerRow) {
    colByHeader = {};
    if (!Array.isArray(headerRow)) {
        return;
    }
    for (let i = 0; i < headerRow.length; i++) {
        const h = String(headerRow[i] ?? "").trim();
        if (h) {
            colByHeader[h] = i;
        }
    }
}

function colIndex(headerName) {
    return Object.prototype.hasOwnProperty.call(colByHeader, headerName) ? colByHeader[headerName] : -1;
}

function headerNameForIndex(colIdx) {
    for (const name in colByHeader) {
        if (colByHeader[name] === colIdx) {
            return name;
        }
    }
    return "";
}

function isMediaHeader(headerName) {
    return MEDIA_HEADERS.has(headerName);
}

function buildPlotImgCell(thumbSrc, detailSrc, altText) {
    const thumb = `${thumbSrc}?v=${ASSET_CACHE_BUST}`;
    const href = detailSrc || thumbSrc;
    return `<a href="${href}" target="_blank" rel="noopener noreferrer"><img src="${thumb}" alt="${altText}" onerror="this.closest('a').outerHTML='N/A';"></a>`;
}

function buildHbetaCellHtml(gaiaId) {
    if (!gaiaId) {
        return "N/A";
    }
    const apfSrc = `stars/Gaia_DR3_${gaiaId}/Gaia/Plots/Gaia_DR3_${gaiaId}_28_hbeta.png`;
    const kpfSrc = `stars/Gaia_DR3_${gaiaId}/Keck/Plots/Gaia_DR3_${gaiaId}_kpf_hbeta.png`;
    const hasKeck = KECK_GAIA_IDS.has(gaiaId);
    if (!hasKeck) {
        return buildPlotImgCell(apfSrc, apfSrc, "H-beta plot");
    }
    const thumb = `${apfSrc}?v=${ASSET_CACHE_BUST}`;
    return `<a href="${apfSrc}" target="_blank" rel="noopener noreferrer" data-apf="${apfSrc}" data-kpf="${kpfSrc}"><img src="${thumb}" alt="H-beta plot" onerror="(function(img){var a=img.closest('a');if(!a){return;}if(!a.dataset.fallbackTried){a.dataset.fallbackTried='1';var kpf=a.getAttribute('data-kpf');if(kpf){a.href=kpf;img.src=kpf+'?v=${ASSET_CACHE_BUST}';return;}}a.outerHTML='N/A';})(this);"></a>`;
}

function buildApfRvPlotCellHtml(gaiaId) {
    if (!gaiaId) {
        return "N/A";
    }
    const src = `stars/Gaia_DR3_${gaiaId}/Gaia/Plots/Gaia_DR3_${gaiaId}_rv_plot.png`;
    return buildPlotImgCell(src, src, "APF RV plot");
}

function buildKeckRvCellHtml(gaiaId) {
    if (!gaiaId) {
        return "N/A";
    }
    const src = `stars/Gaia_DR3_${gaiaId}/Keck/Plots/${gaiaId}_kpf_rv_plot.png`;
    return buildPlotImgCell(src, src, "KPF RV plot");
}

function buildRvFitCellHtml(gaiaId) {
    if (!gaiaId) {
        return "N/A";
    }
    const thumb = `stars/Gaia_DR3_${gaiaId}/Gaia/RV_Fit/${gaiaId}_keplerian_fit.png`;
    const detail = `stars/Gaia_DR3_${gaiaId}/Gaia/Plots/Gaia_DR3_${gaiaId}_keplerian_residuals.png`;
    return buildPlotImgCell(thumb, detail, "APF RV fit");
}

function syncUrlState() {
    if (suppressUrlSync) {
        return;
    }
    const p = new URLSearchParams();
    const starInput = document.getElementById("STARinput");
    if (starInput && starInput.value.trim() !== "") {
        p.set("q", starInput.value.trim());
    }
    p.set("rows", String(pageSize));
    p.set("page", String(currentPage));
    if (sortColumn !== null) {
        p.set("sort", String(sortColumn));
        p.set("dir", sortDirection);
    }

    const rv = document.getElementById("toggle-rv");
    const flux = document.getElementById("toggle-flux");
    const source = document.getElementById("toggle-source");
    const apf = document.getElementById("toggle-apf-data");
    const keck = document.getElementById("toggle-keck");
    const recentApf = document.getElementById("toggle-recent-apf");
    const recentKeck = document.getElementById("toggle-recent-keck");
    const recentApfWeek = document.getElementById("toggle-recent-apf-week");
    const recentKeckWeek = document.getElementById("toggle-recent-keck-week");
    if (rv && rv.checked) p.set("rv", "1");
    if (flux && flux.checked) p.set("flux", "1");
    if (source && source.checked) p.set("source", "1");
    if (apf && apf.checked) p.set("apf", "1");
    if (keck && keck.checked) p.set("keck", "1");
    if (recentApf && recentApf.checked) p.set("rapf", "1");
    if (recentKeck && recentKeck.checked) p.set("rkeck", "1");
    if (recentApfWeek && recentApfWeek.checked) p.set("w7apf", "1");
    if (recentKeckWeek && recentKeckWeek.checked) p.set("w7keck", "1");
    for (let i = 0; i < YES_NO_PARAMS.length; i++) {
        const filterValue = getParamFilterValue(YES_NO_PARAMS[i]);
        if (filterValue) {
            p.set(YES_NO_PARAMS[i].urlKey, filterValue);
        }
    }

    const rangeKeys = [
        ["RA", "ra"], ["DEC", "dec"], ["PARALLAX", "parallax"],
        ["PMRA", "pmra"], ["PMDEC", "pmdec"], ["PERIOD", "period"],
        ["ECCENTRICITY", "ecc"], ["M1", "m1"], ["M2", "m2"]
    ];
    for (let i = 0; i < rangeKeys.length; i++) {
        const label = rangeKeys[i][0];
        const short = rangeKeys[i][1];
        const start = document.getElementById(`${label}_start`);
        const end = document.getElementById(`${label}_end`);
        if (start && start.value.trim() !== "") p.set(`${short}_min`, start.value.trim());
        if (end && end.value.trim() !== "") p.set(`${short}_max`, end.value.trim());
    }

    const qs = p.toString();
    const newUrl = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
    history.replaceState(null, "", newUrl);
}

function parseCellNumber(cell) {
    if (!cell) {
        return NaN;
    }
    const n = parseFloat(cell.textContent.trim());
    return Number.isNaN(n) ? NaN : n;
}

function normalizeGaiaId(text) {
    const match = String(text || "").match(/(\d{8,})/);
    return match && match[1] ? match[1] : "";
}

function cleanKeckValue(v) {
    const s = String(v || "").trim();
    if (!s || s.toUpperCase() === "N/A" || s === "--") {
        return "--";
    }
    return s;
}

function loadOptionalKeckCatalog() {
    return fetch("tables/keck_targets.csv", { cache: "no-store" })
        .then(response => {
            if (!response.ok) {
                return "";
            }
            return response.text();
        })
        .then(csvText => {
            if (!csvText) {
                return;
            }
            const lines = csvText.split("\n");
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line || line.startsWith("#")) {
                    continue;
                }
                const firstCol = line.split(",")[0];
                const gaiaId = normalizeGaiaId(firstCol);
                if (gaiaId) {
                    KECK_GAIA_IDS.add(gaiaId);
                    const parts = line.split(",");
                    const ra = cleanKeckValue(parts[1]);
                    const dec = cleanKeckValue(parts[2]);
                    const parallax = cleanKeckValue(parts[3]);
                    const pmra = cleanKeckValue(parts[4]);
                    const pmdec = cleanKeckValue(parts[5]);
                    const period = cleanKeckValue(parts[6]);
                    const ecc = cleanKeckValue(parts[7]);
                    const m1 = cleanKeckValue(parts[8]);
                    const m2 = cleanKeckValue(parts[9]);
                    KECK_TARGETS.set(gaiaId, { ra, dec, parallax, pmra, pmdec, period, ecc, m1, m2 });
                }
            }
        })
        .catch(() => {});
}

function mergeKeckOnlyRows(tableData) {
    if (!Array.isArray(tableData) || tableData.length === 0) {
        return tableData;
    }
    const header = tableData[0];
    const colCount = header.length;
    const gaiaCol = 0;
    const raCol = 1;
    const decCol = 2;
    const parallaxCol = 3;
    const pmraCol = 4;
    const pmdecCol = 5;
    const periodCol = 6;
    const eccCol = 7;
    const m1Col = 8;
    const m2Col = 9;
    const dataProductsCol = 14;

    const existingIds = new Set();
    for (let i = 1; i < tableData.length; i++) {
        const row = tableData[i];
        if (!row) {
            continue;
        }
        const gaiaId = normalizeGaiaId(row[gaiaCol]);
        if (gaiaId) {
            existingIds.add(gaiaId);
        }
    }

    KECK_TARGETS.forEach((info, gaiaId) => {
        if (existingIds.has(gaiaId)) {
            return;
        }
        const row = new Array(colCount).fill("N/A");
        row[gaiaCol] = `Gaia DR3 ${gaiaId}`;
        row[raCol] = info.ra || "N/A";
        row[decCol] = info.dec || "N/A";
        row[parallaxCol] = info.parallax || "--";
        row[pmraCol] = info.pmra || "--";
        row[pmdecCol] = info.pmdec || "--";
        row[periodCol] = info.period || "--";
        row[eccCol] = info.ecc || "--";
        row[m1Col] = info.m1 || "--";
        row[m2Col] = info.m2 || "--";
        row[dataProductsCol] = `<a href='stars/Gaia_DR3_${gaiaId}'> ${gaiaId} </a>`;
        tableData.push(row);
    });

    return tableData;
}

function extractHrefFromAnchorHtml(text) {
    if (typeof text !== "string") {
        return "";
    }
    const hrefMatch = text.match(/href=['"]([^'"]+)['"]/i);
    return (hrefMatch && hrefMatch[1]) ? hrefMatch[1].replace(/\/+$/, "") : "";
}

function toCsvCell(text) {
    const s = String(text ?? "");
    if (s.includes(",") || s.includes("\"") || s.includes("\n")) {
        return `"${s.replaceAll("\"", "\"\"")}"`;
    }
    return s;
}

function hasMedia(cell) {
    if (!cell) {
        return false;
    }
    const html = cell.innerHTML.toLowerCase();
    if (html.includes("<img") || html.includes("<a ")) {
        return true;
    }
    const txt = cell.textContent.trim().toUpperCase();
    return !!txt && txt !== "N/A";
}

function getDataRows() {
    return Array.from(table.getElementsByTagName("tr")).slice(1);
}

function anyRangeQueryActive() {
    for (let i = 0; i < headers_array.length; i++) {
        const start = document.getElementById(`${headers_array[i]}_start`);
        const end = document.getElementById(`${headers_array[i]}_end`);
        if (!start || !end) {
            continue;
        }
        if (start.value.trim() !== "" || end.value.trim() !== "") {
            return true;
        }
    }
    return false;
}

function anyToggleActive() {
    const rv = document.getElementById("toggle-rv");
    const flux = document.getElementById("toggle-flux");
    const source = document.getElementById("toggle-source");
    const apf = document.getElementById("toggle-apf-data");
    const recent = document.getElementById("toggle-recent-apf");
    const keck = document.getElementById("toggle-keck");
    const recentKeck = document.getElementById("toggle-recent-keck");
    const recentApfWeek = document.getElementById("toggle-recent-apf-week");
    const recentKeckWeek = document.getElementById("toggle-recent-keck-week");
    for (let i = 0; i < YES_NO_PARAMS.length; i++) {
        if (getParamFilterValue(YES_NO_PARAMS[i])) {
            return true;
        }
    }
    return (rv && rv.checked) || (flux && flux.checked) || (source && source.checked) || (apf && apf.checked) || (recent && recent.checked) || (keck && keck.checked) || (recentKeck && recentKeck.checked) || (recentApfWeek && recentApfWeek.checked) || (recentKeckWeek && recentKeckWeek.checked);
}

function hasActiveQuery() {
    const starInput = document.getElementById("STARinput");
    return (starInput && starInput.value.trim() !== "") || anyRangeQueryActive() || anyToggleActive();
}

function rowMatchesBaseFilters(row) {
    const cells = row.getElementsByTagName("td");

    const starInput = document.getElementById("STARinput");
    const starFilter = starInput ? starInput.value.trim().toUpperCase() : "";
    if (starFilter) {
        const txt = cells[0] ? cells[0].textContent.toUpperCase() : "";
        if (!txt.includes(starFilter)) {
            return false;
        }
    }

    for (let i = 0; i < headers_array.length; i++) {
        const start = document.getElementById(`${headers_array[i]}_start`);
        const end = document.getElementById(`${headers_array[i]}_end`);
        if (!start || !end) {
            continue;
        }

        const val = parseCellNumber(cells[i + 1]);
        const hasStart = start.value.trim() !== "";
        const hasEnd = end.value.trim() !== "";

        if ((hasStart || hasEnd) && Number.isNaN(val)) {
            return false;
        }

        if (hasStart && val < parseFloat(start.value)) {
            return false;
        }
        if (hasEnd && val > parseFloat(end.value)) {
            return false;
        }
    }

    return true;
}

function rowMatchesFilters(row) {
    if (!rowMatchesBaseFilters(row)) {
        return false;
    }
    const cells = row.getElementsByTagName("td");
    const rvToggle = document.getElementById("toggle-rv");
    const fluxToggle = document.getElementById("toggle-flux");
    const sourceToggle = document.getElementById("toggle-source");
    const apfToggle = document.getElementById("toggle-apf-data");
    const recentApfToggle = document.getElementById("toggle-recent-apf");
    const keckToggle = document.getElementById("toggle-keck");
    const recentKeckToggle = document.getElementById("toggle-recent-keck");
    const recentApfWeekToggle = document.getElementById("toggle-recent-apf-week");
    const recentKeckWeekToggle = document.getElementById("toggle-recent-keck-week");

    if (rvToggle && rvToggle.checked && !hasMedia(cells[10])) {
        return false;
    }
    if (fluxToggle && fluxToggle.checked && !hasMedia(cells[12])) {
        return false;
    }
    if (sourceToggle && sourceToggle.checked && !hasMedia(cells[13])) {
        return false;
    }
    if (apfToggle && apfToggle.checked && row.dataset.hasApf !== "1") {
        return false;
    }
    if (recentApfToggle && recentApfToggle.checked && row.dataset.apfRecent !== "1") {
        return false;
    }
    if (recentApfWeekToggle && recentApfWeekToggle.checked && row.dataset.apfRecentWeek !== "1") {
        return false;
    }
    if (keckToggle && keckToggle.checked && row.dataset.hasKeck !== "1") {
        return false;
    }
    if (recentKeckToggle && recentKeckToggle.checked && row.dataset.keckRecent !== "1") {
        return false;
    }
    if (recentKeckWeekToggle && recentKeckWeekToggle.checked && row.dataset.keckRecentWeek !== "1") {
        return false;
    }
    for (let i = 0; i < YES_NO_PARAMS.length; i++) {
        const selected = getParamFilterValue(YES_NO_PARAMS[i]);
        if (selected && row.dataset[YES_NO_PARAMS[i].datasetKey] !== selected) {
            return false;
        }
    }
    return true;
}

function colorRows(rows) {
    let visibleCount = 0;
    for (let i = 1; i < rows.length; i++) {
        if (rows[i].style.display === "none") {
            continue;
        }
        visibleCount += 1;
        rows[i].style.backgroundColor = (visibleCount % 2 === 0) ? "#e6edf5" : "#ffffff";
    }
}

function updatePageStatus(totalPages, selected) {
    if (page_status) {
        page_status.innerHTML = `Page ${currentPage}/${totalPages} (${selected} selected)`;
    }
}

function updateToggleCounts(baseRows) {
    const rvCount = document.getElementById("count-rv");
    const fluxCount = document.getElementById("count-flux");
    const sourceCount = document.getElementById("count-source");
    const apfCount = document.getElementById("count-apf-data");
    const recentCount = document.getElementById("count-recent-apf");
    const keckCount = document.getElementById("count-keck");
    const recentKeckCount = document.getElementById("count-recent-keck");
    const recentApfWeekCount = document.getElementById("count-recent-apf-week");
    const recentKeckWeekCount = document.getElementById("count-recent-keck-week");
    const yesNoCounts = YES_NO_PARAMS.map(param => document.getElementById(param.countId));
    let rv = 0;
    let flux = 0;
    let source = 0;
    let apf = 0;
    let recent = 0;
    let keck = 0;
    let recentKeck = 0;
    let recentApfWeek = 0;
    let recentKeckWeek = 0;
    const yesNoTotals = YES_NO_PARAMS.map(() => ({ Y: 0, N: 0 }));
    for (let i = 0; i < baseRows.length; i++) {
        const cells = baseRows[i].getElementsByTagName("td");
        if (hasMedia(cells[10])) rv += 1;
        if (hasMedia(cells[12])) flux += 1;
        if (hasMedia(cells[13])) source += 1;
        if (baseRows[i].dataset.hasApf === "1") apf += 1;
        if (baseRows[i].dataset.apfRecent === "1") recent += 1;
        if (baseRows[i].dataset.apfRecentWeek === "1") recentApfWeek += 1;
        if (baseRows[i].dataset.hasKeck === "1") keck += 1;
        if (baseRows[i].dataset.keckRecent === "1") recentKeck += 1;
        if (baseRows[i].dataset.keckRecentWeek === "1") recentKeckWeek += 1;
        for (let j = 0; j < YES_NO_PARAMS.length; j++) {
            const v = normalizeYesNoValue(baseRows[i].dataset[YES_NO_PARAMS[j].datasetKey]);
            yesNoTotals[j][v] += 1;
        }
    }
    if (rvCount) rvCount.textContent = `(${rv})`;
    if (fluxCount) fluxCount.textContent = `(${flux})`;
    if (sourceCount) sourceCount.textContent = `(${source})`;
    if (apfCount) apfCount.textContent = `(${apf})`;
    if (recentCount) recentCount.textContent = `(${recent})`;
    if (keckCount) keckCount.textContent = `(${keck})`;
    if (recentKeckCount) recentKeckCount.textContent = `(${recentKeck})`;
    if (recentApfWeekCount) recentApfWeekCount.textContent = `(${recentApfWeek})`;
    if (recentKeckWeekCount) recentKeckWeekCount.textContent = `(${recentKeckWeek})`;
    for (let i = 0; i < yesNoCounts.length; i++) {
        if (yesNoCounts[i]) {
            yesNoCounts[i].textContent = `(Y:${yesNoTotals[i].Y} N:${yesNoTotals[i].N})`;
        }
    }
}

function renderDataProductsCell(row) {
    const base = row.dataset.starBase || "";
    if (!base) {
        return "N/A";
    }
    const apfUrl = `${base}/Gaia/`;
    const swiftUrl = `${base}/Swift/`;
    const keckUrl = `${base}/Keck/`;
    const hasApf = row.dataset.hasApf !== "0";
    const hasSwift = row.dataset.hasSwift === "1";
    const hasKeck = row.dataset.hasKeck === "1";
    const apfRecent = row.dataset.apfRecent || "unknown";
    const apfChecked = row.dataset.apfChecked === "1";
    const apfAgeDays = parseFloat(row.dataset.apfAgeDays || "");
    const keckRecent = row.dataset.keckRecent || "unknown";
    const keckChecked = row.dataset.keckChecked === "1";
    const keckAgeDays = parseFloat(row.dataset.keckAgeDays || "");
    let apfStateClass = "apf-unknown";
    let apfStateText = "";
    if (apfChecked && Number.isFinite(apfAgeDays)) {
        if (apfRecent === "1") {
            apfStateClass = "apf-recent";
        } else if (apfRecent === "0") {
            apfStateClass = "apf-old";
        }
        apfStateText = `last epoch ${Math.round(apfAgeDays)} d ago`;
    }
    let keckStateClass = "apf-unknown";
    let keckStateText = "";
    if (keckChecked && Number.isFinite(keckAgeDays)) {
        if (keckRecent === "1") {
            keckStateClass = "apf-recent";
        } else if (keckRecent === "0") {
            keckStateClass = "apf-old";
        }
        keckStateText = `last epoch ${Math.round(keckAgeDays)} d ago`;
    }

    let html = `<div class="product-links">`;
    if (hasApf) {
        html += `<div class="product-line"><a href="${apfUrl}" target="_blank" rel="noopener noreferrer">APF</a><span class="apf-state ${apfStateClass}">${apfStateText}</span></div>`;
    }
    if (hasKeck) {
        html += `<div class="product-line"><a href="${keckUrl}" target="_blank" rel="noopener noreferrer">KPF</a><span class="apf-state ${keckStateClass}">${keckStateText}</span></div>`;
    }
    if (hasSwift) {
        html += `<div class="product-line"><a href="${swiftUrl}" target="_blank" rel="noopener noreferrer">Swift</a></div>`;
    }
    html += `</div>`;
    return html;
}

function mjdNow() {
    return Date.now() / 86400000 + 40587;
}

function isPlausibleMjd(mjd, nowMjd) {
    return Number.isFinite(mjd) && mjd > 40000 && mjd <= (nowMjd + 1);
}

function extractLatestMjd(summaryText) {
    const lines = String(summaryText || "").split("\n");
    let latest = null;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line || line.startsWith("#")) {
            continue;
        }
        const parts = line.split(/\s+/);
        if (parts.length < 5) {
            continue;
        }
        const mjd = parseFloat(parts[1]);
        if (!Number.isNaN(mjd) && (latest === null || mjd > latest)) {
            latest = mjd;
        }
    }
    return latest;
}

function extractEpochCount(summaryText) {
    const lines = String(summaryText || "").split("\n");
    const uniqueEpochs = new Set();
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line || line.startsWith("#")) {
            continue;
        }
        const parts = line.split(/\s+/);
        if (parts.length >= 5) {
            // Count unique APF epochs only once even if the summary contains
            // duplicate rows with different path prefixes.
            const inputPath = parts[0] || "";
            const baseName = inputPath.split("/").pop() || inputPath;
            const mjd = parts[1] || "";
            uniqueEpochs.add(`${baseName}|${mjd}`);
        }
    }
    return uniqueEpochs.size;
}

async function updateApfRecencyFlags() {
    const rows = getDataRows();
    const nowMjd = mjdNow();
    const tasks = rows.map(async (row) => {
        const gaiaId = row.dataset.gaiaId;
        if (!gaiaId || row.dataset.hasApf === "0") {
            return;
        }
        const summaryUrl = `stars/Gaia_DR3_${gaiaId}/Gaia/${gaiaId}_summary.txt`;
        try {
            const resp = await fetch(summaryUrl, { cache: "no-store" });
            if (!resp.ok) {
                row.dataset.apfRecent = "unknown";
                row.dataset.apfRecentWeek = "unknown";
                row.dataset.apfAgeDays = "";
                row.dataset.apfChecked = "1";
                return;
            }
            const txt = await resp.text();
            const latestMjd = extractLatestMjd(txt);
            const epochCount = extractEpochCount(txt);
            row.dataset.apfCount = String(epochCount);
            if (latestMjd === null || !isPlausibleMjd(latestMjd, nowMjd)) {
                row.dataset.apfRecent = "unknown";
                row.dataset.apfRecentWeek = "unknown";
                row.dataset.apfAgeDays = "";
                row.dataset.apfChecked = "1";
                return;
            }
            const ageDays = nowMjd - latestMjd;
            row.dataset.apfRecent = (ageDays >= 0 && ageDays <= RECENT_APF_DAYS) ? "1" : "0";
            row.dataset.apfRecentWeek = (ageDays >= 0 && ageDays <= RECENT_WEEK_DAYS) ? "1" : "0";
            row.dataset.apfLatestMjd = String(latestMjd);
            row.dataset.apfAgeDays = String(ageDays);
            row.dataset.apfChecked = "1";
        } catch {
            row.dataset.apfRecent = "unknown";
            row.dataset.apfRecentWeek = "unknown";
            row.dataset.apfAgeDays = "";
            row.dataset.apfChecked = "1";
            row.dataset.apfCount = row.dataset.apfCount || "0";
        }
    });
    await Promise.allSettled(tasks);

    for (let i = 0; i < rows.length; i++) {
        const cell = rows[i].getElementsByTagName("td")[14];
        if (cell) {
            cell.innerHTML = renderDataProductsCell(rows[i]);
        }
    }
    applyFiltersAndPagination();
}

function sortRowsByApfCount() {
    const rows = getDataRows();
    if (rows.length === 0) {
        return;
    }

    sortColumn = null;

    rows.sort((a, b) => {
        const av = parseInt(a.dataset.apfCount || "0", 10);
        const bv = parseInt(b.dataset.apfCount || "0", 10);
        const an = Number.isNaN(av) ? 0 : av;
        const bn = Number.isNaN(bv) ? 0 : bv;
        const cmp = an - bn;
        return apfCountSortDirection === "asc" ? cmp : -cmp;
    });

    for (let i = 0; i < rows.length; i++) {
        table.appendChild(rows[i]);
    }

    const btn = document.getElementById("sort-apf-count");
    if (btn) {
        btn.textContent = apfCountSortDirection === "desc" ? "Sort by APF points ↑" : "Sort by APF points ↓";
    }
    apfCountSortDirection = apfCountSortDirection === "desc" ? "asc" : "desc";

    applyFiltersAndPagination();
}

function sortRowsByKeckCount() {
    const rows = getDataRows();
    if (rows.length === 0) {
        return;
    }

    sortColumn = null;

    rows.sort((a, b) => {
        const av = parseInt(a.dataset.keckCount || "0", 10);
        const bv = parseInt(b.dataset.keckCount || "0", 10);
        const an = Number.isNaN(av) ? 0 : av;
        const bn = Number.isNaN(bv) ? 0 : bv;
        const cmp = an - bn;
        return keckCountSortDirection === "asc" ? cmp : -cmp;
    });

    for (let i = 0; i < rows.length; i++) {
        table.appendChild(rows[i]);
    }

    const btn = document.getElementById("sort-kpf-count");
    if (btn) {
        btn.textContent = keckCountSortDirection === "desc" ? "Sort by KPF points ↑" : "Sort by KPF points ↓";
    }
    keckCountSortDirection = keckCountSortDirection === "desc" ? "asc" : "desc";

    applyFiltersAndPagination();
}

function parseAgeForSort(value) {
    const v = parseFloat(value || "");
    return Number.isFinite(v) ? v : Number.POSITIVE_INFINITY;
}

function sortRowsByApfRecent() {
    const rows = getDataRows();
    if (rows.length === 0) {
        return;
    }

    sortColumn = null;

    rows.sort((a, b) => {
        const av = parseAgeForSort(a.dataset.apfAgeDays);
        const bv = parseAgeForSort(b.dataset.apfAgeDays);
        const cmp = av - bv;
        return apfRecentSortDirection === "asc" ? cmp : -cmp;
    });

    for (let i = 0; i < rows.length; i++) {
        table.appendChild(rows[i]);
    }

    const btn = document.getElementById("sort-apf-recent");
    if (btn) {
        btn.textContent = apfRecentSortDirection === "asc" ? "Sort by recent APF ↑" : "Sort by recent APF ↓";
    }
    apfRecentSortDirection = apfRecentSortDirection === "asc" ? "desc" : "asc";

    applyFiltersAndPagination();
}

function sortRowsByKeckRecent() {
    const rows = getDataRows();
    if (rows.length === 0) {
        return;
    }

    sortColumn = null;

    rows.sort((a, b) => {
        const av = parseAgeForSort(a.dataset.keckAgeDays);
        const bv = parseAgeForSort(b.dataset.keckAgeDays);
        const cmp = av - bv;
        return keckRecentSortDirection === "asc" ? cmp : -cmp;
    });

    for (let i = 0; i < rows.length; i++) {
        table.appendChild(rows[i]);
    }

    const btn = document.getElementById("sort-kpf-recent");
    if (btn) {
        btn.textContent = keckRecentSortDirection === "asc" ? "Sort by recent KPF ↑" : "Sort by recent KPF ↓";
    }
    keckRecentSortDirection = keckRecentSortDirection === "asc" ? "desc" : "asc";

    applyFiltersAndPagination();
}

async function updateKeckRecencyFlags() {
    const rows = getDataRows();
    const nowMjd = mjdNow();
    const tasks = rows.map(async (row) => {
        const gaiaId = row.dataset.gaiaId;
        if (!gaiaId || row.dataset.hasKeck !== "1") {
            return;
        }
        const candidates = [
            `stars/Gaia_DR3_${gaiaId}/Keck/${gaiaId}_summary.txt`,
            `stars/Gaia_DR3_${gaiaId}/Keck/Gaia_DR3_${gaiaId}_summary.txt`
        ];
        let latestMjd = null;
        let chosenSummaryText = "";
        for (let i = 0; i < candidates.length; i++) {
            try {
                const resp = await fetch(candidates[i], { cache: "no-store" });
                if (!resp.ok) {
                    continue;
                }
                const txt = await resp.text();
                latestMjd = extractLatestMjd(txt);
                if (latestMjd !== null) {
                    chosenSummaryText = txt;
                    break;
                }
            } catch {
                // try next candidate
            }
        }
        if (latestMjd === null || !isPlausibleMjd(latestMjd, nowMjd)) {
            row.dataset.keckRecent = "unknown";
            row.dataset.keckRecentWeek = "unknown";
            row.dataset.keckAgeDays = "";
            row.dataset.keckChecked = "1";
            row.dataset.keckCount = row.dataset.keckCount || "0";
            return;
        }
        row.dataset.keckCount = String(extractEpochCount(chosenSummaryText));
        const ageDays = nowMjd - latestMjd;
        row.dataset.keckRecent = (ageDays >= 0 && ageDays <= RECENT_KECK_DAYS) ? "1" : "0";
        row.dataset.keckRecentWeek = (ageDays >= 0 && ageDays <= RECENT_WEEK_DAYS) ? "1" : "0";
        row.dataset.keckAgeDays = String(ageDays);
        row.dataset.keckChecked = "1";
    });
    await Promise.allSettled(tasks);

    for (let i = 0; i < rows.length; i++) {
        const cell = rows[i].getElementsByTagName("td")[14];
        if (cell) {
            cell.innerHTML = renderDataProductsCell(rows[i]);
        }
    }
    applyFiltersAndPagination();
}

function applyFiltersAndPagination() {
    const rows = getDataRows();
    const baseMatched = [];
    const matched = [];

    for (let i = 0; i < rows.length; i++) {
        if (rowMatchesBaseFilters(rows[i])) {
            baseMatched.push(rows[i]);
        }
        if (rowMatchesFilters(rows[i])) {
            matched.push(rows[i]);
        }
    }
    lastMatchedRows = matched.slice();
    updateToggleCounts(baseMatched);

    const perPage = pageSize === "all" ? matched.length : parseInt(pageSize, 10);
    const totalPages = Math.max(1, pageSize === "all" ? 1 : Math.ceil(matched.length / perPage));
    if (currentPage > totalPages) {
        currentPage = totalPages;
    }

    for (let i = 0; i < rows.length; i++) {
        rows[i].style.display = "none";
    }

    if (pageSize === "all") {
        for (let i = 0; i < matched.length; i++) {
            matched[i].style.display = "";
        }
    } else {
        const startIdx = (currentPage - 1) * perPage;
        const endIdx = startIdx + perPage;
        for (let i = startIdx; i < endIdx && i < matched.length; i++) {
            matched[i].style.display = "";
        }
    }

    colorRows(table.rows);
    updatePageStatus(totalPages, matched.length);
    syncUrlState();
}

function updateSortIndicators() {
    const headers = table.getElementsByTagName("th");
    for (let i = 0; i < headers.length; i++) {
        const base = headers[i].dataset.baseLabel || headers[i].textContent.replace(/\s*[▲▼]$/, "");
        headers[i].dataset.baseLabel = base;
        headers[i].innerHTML = base;
        if (i === sortColumn) {
            headers[i].innerHTML = `${base} ${sortDirection === "asc" ? "▲" : "▼"}`;
        }
    }
}

function prettifyHeaderText(rawText) {
    if (typeof rawText !== "string") {
        return rawText;
    }
    return headerDisplayMap[rawText] || rawText;
}

function sortTable(n) {
    if (sortColumn === n) {
        sortDirection = sortDirection === "asc" ? "desc" : "asc";
    } else {
        sortColumn = n;
        sortDirection = "asc";
    }

    sortRowsByCurrent();
    applyFiltersAndPagination();
}

function sortRowsByCurrent() {
    const rows = getDataRows();
    if (rows.length === 0 || sortColumn === null) {
        return;
    }

    rows.sort((a, b) => {
        const ax = a.getElementsByTagName("td")[sortColumn];
        const bx = b.getElementsByTagName("td")[sortColumn];
        const av = ax ? ax.textContent.trim() : "";
        const bv = bx ? bx.textContent.trim() : "";
        const an = parseFloat(av);
        const bn = parseFloat(bv);

        let cmp = 0;
        if (!Number.isNaN(an) && !Number.isNaN(bn)) {
            cmp = an - bn;
        } else {
            cmp = av.localeCompare(bv);
        }

        return sortDirection === "asc" ? cmp : -cmp;
    });

    for (let i = 0; i < rows.length; i++) {
        table.appendChild(rows[i]);
    }

    updateSortIndicators();
}

function arrayToTable(tableData) {
    table.innerHTML = "";
    let stripes = 0;
    const yesNoColumnsByIndex = new Map();
    if (Array.isArray(tableData) && Array.isArray(tableData[0])) {
        const headerRow = tableData[0];
        rebuildColumnIndex(headerRow);
        for (let i = 0; i < YES_NO_PARAMS.length; i++) {
            const colIdx = headerRow.indexOf(YES_NO_PARAMS[i].header);
            if (colIdx >= 0) {
                yesNoColumnsByIndex.set(colIdx, YES_NO_PARAMS[i]);
            }
        }
    }

    for (let i = 0; i < tableData.length; i++) {
        const rowData = tableData[i];
        if (!rowData || Object.keys(rowData).length === 0) {
            continue;
        }

        const row = document.createElement("tr");
        row.style.backgroundColor = stripes === 0 ? "#e6edf5" : "#ffffff";
        stripes = stripes === 0 ? 1 : 0;

        for (const key in rowData) {
            let cell;
            if (i === 0) {
                cell = document.createElement("th");
                const col = parseInt(key, 10);
                cell.onclick = function () {
                    sortTable(col);
                };
                cell.style.cursor = "pointer";
            } else {
                cell = document.createElement("td");
            }

            let text = rowData[key];
            const colIdx = parseInt(key, 10);
            if (i === 0) {
                text = prettifyHeaderText(text);
                cell.dataset.baseLabel = text;
            }
            if (i > 0 && colIdx === 0) {
                const gaiaId = normalizeGaiaId(text);
                if (gaiaId) {
                    row.dataset.gaiaId = gaiaId;
                }
                text = buildGaiaNameCellHtml(text);
            }
            if (i > 0 && yesNoColumnsByIndex.has(colIdx)) {
                const param = yesNoColumnsByIndex.get(colIdx);
                const normalized = normalizeYesNoValue(text);
                text = normalized;
                row.dataset[param.datasetKey] = normalized;
            }
            const headerName = i === 0 ? String(text ?? "").trim() : headerNameForIndex(colIdx);
            if (
                i > 0
                && headerName
                && !isMediaHeader(headerName)
                && typeof text === "string"
                && /<img\b/i.test(text)
            ) {
                text = "";
            }
            if (i > 0 && colIdx !== 0 && !isMediaHeader(headerName) && !Number.isNaN(Number(text))) {
                text = Number(text).toFixed(5);
            }
            if (i > 0 && isMediaHeader(headerName) && hasNonNAContent(text)) {
                text = buildMediaCellHtml(text);
            }
            const rvPlotCol = colIndex("RV PLOT");
            const rvFitCol = colIndex("RV FIT");
            const fluxCol = colIndex("FLUX PLOT");
            const sourceCol = colIndex("SOURCE IMAGE");
            const dataProductsCol = colIndex("DATA PRODUCTS");
            if (i > 0 && colIdx === rvPlotCol) {
                cell.classList.add("col-rv-plot");
                const gaiaId = row.dataset.gaiaId || normalizeGaiaId(rowData[String(colIndex("GAIA NAME"))] ?? rowData["0"]);
                text = buildApfRvPlotCellHtml(gaiaId);
                if (!hasNonNAContent(text) && KECK_GAIA_IDS.has(gaiaId)) {
                    text = buildKeckRvCellHtml(gaiaId);
                }
            }
            if (i > 0 && colIdx === sourceCol) {
                cell.classList.add("col-source-image");
            }
            if (i > 0 && colIdx === fluxCol) {
                cell.classList.add("col-flux-plot");
                const gaiaId = row.dataset.gaiaId || normalizeGaiaId(rowData[String(colIndex("GAIA NAME"))] ?? rowData["0"]);
                text = buildHbetaCellHtml(gaiaId);
            }
            if (i > 0 && colIdx === rvFitCol) {
                cell.classList.add("col-rv-fit");
                const gaiaId = row.dataset.gaiaId || normalizeGaiaId(rowData[String(colIndex("GAIA NAME"))] ?? rowData["0"]);
                text = buildRvFitCellHtml(gaiaId);
            }
            if (i > 0 && colIdx === dataProductsCol) {
                const base = extractHrefFromAnchorHtml(text);
                const gaiaId = row.dataset.gaiaId || "";
                row.dataset.starBase = gaiaId ? `stars/Gaia_DR3_${gaiaId}` : base;
                const rvPlotCell = rvPlotCol >= 0 ? rowData[String(rvPlotCol)] : "";
                const rvFitCell = rvFitCol >= 0 ? rowData[String(rvFitCol)] : "";
                const fluxCell = fluxCol >= 0 ? rowData[String(fluxCol)] : "";
                const sourceCell = sourceCol >= 0 ? rowData[String(sourceCol)] : "";
                row.dataset.hasApf = (hasNonNAContent(rvPlotCell) || hasNonNAContent(rvFitCell)) ? "1" : "0";
                row.dataset.hasSwift = (hasNonNAContent(fluxCell) || hasNonNAContent(sourceCell)) ? "1" : "0";
                row.dataset.hasKeck = KECK_GAIA_IDS.has(row.dataset.gaiaId || "") ? "1" : "0";
                row.dataset.apfRecent = "unknown";
                row.dataset.apfRecentWeek = "unknown";
                row.dataset.apfAgeDays = "";
                row.dataset.apfChecked = "0";
                row.dataset.apfCount = "0";
                row.dataset.keckRecent = "unknown";
                row.dataset.keckRecentWeek = "unknown";
                row.dataset.keckAgeDays = "";
                row.dataset.keckChecked = "0";
                row.dataset.keckCount = "0";
                text = renderDataProductsCell(row);
            }

            cell.innerHTML = text;
            row.appendChild(cell);
        }

        if (i > 0 && row.textContent.trim() === "") {
            continue;
        }
        if (i > 0) {
            for (let j = 0; j < YES_NO_PARAMS.length; j++) {
                if (!row.dataset[YES_NO_PARAMS[j].datasetKey]) {
                    row.dataset[YES_NO_PARAMS[j].datasetKey] = "N";
                }
            }
        }

        table.appendChild(row);
    }

    if (sortColumn !== null) {
        sortRowsByCurrent();
    }
    applyFiltersAndPagination();
    updateApfRecencyFlags();
    updateKeckRecencyFlags();
}

function loadCSV(file) {
    Papa.parse(file, {
        complete: function (results) {
            const normalizedData = ensureYesNoColumns(results.data);
            const tableData = INCLUDE_KECK_ONLY_ROWS ? mergeKeckOnlyRows(normalizedData) : normalizedData;
            arrayToTable(tableData);
        },
        error: function (error) {
            table.innerHTML = "Failed to load table.";
            console.log(error.message);
        }
    });
}

function starSearch() {
    currentPage = 1;
    applyFiltersAndPagination();
}

function searchFunction() {
    currentPage = 1;
    applyFiltersAndPagination();
}

function reset() {
    const inputs = input_container.querySelectorAll("input");
    for (let i = 0; i < inputs.length; i++) {
        inputs[i].value = "";
    }

    const rv = document.getElementById("toggle-rv");
    const flux = document.getElementById("toggle-flux");
    const source = document.getElementById("toggle-source");
    const apf = document.getElementById("toggle-apf-data");
    const keck = document.getElementById("toggle-keck");
    const recentApf = document.getElementById("toggle-recent-apf");
    const recentKeck = document.getElementById("toggle-recent-keck");
    const recentApfWeek = document.getElementById("toggle-recent-apf-week");
    const recentKeckWeek = document.getElementById("toggle-recent-keck-week");
    if (rv) rv.checked = false;
    if (flux) flux.checked = false;
    if (source) source.checked = false;
    if (apf) apf.checked = false;
    if (keck) keck.checked = false;
    if (recentApf) recentApf.checked = false;
    if (recentKeck) recentKeck.checked = false;
    if (recentApfWeek) recentApfWeek.checked = false;
    if (recentKeckWeek) recentKeckWeek.checked = false;
    for (let i = 0; i < YES_NO_PARAMS.length; i++) {
        const control = document.getElementById(YES_NO_PARAMS[i].toggleId);
        if (control) {
            control.value = "";
        }
    }

    currentPage = 1;
    applyFiltersAndPagination();
}

async function copyViewLink(btn) {
    try {
        await navigator.clipboard.writeText(window.location.href);
        if (btn) {
            const bg = btn.style.backgroundColor;
            btn.style.backgroundColor = "#fff4b3";
            setTimeout(() => { btn.style.backgroundColor = bg; }, 900);
        }
    } catch {
        // no-op
    }
}

function downloadSelectedCsv() {
    const headers = Array.from(table.getElementsByTagName("th")).map(h => h.textContent.replace(/\s*[▲▼]$/, "").trim());
    const lines = [];
    lines.push(headers.map(toCsvCell).join(","));
    for (let i = 0; i < lastMatchedRows.length; i++) {
        const row = lastMatchedRows[i];
        const tds = row.getElementsByTagName("td");
        const vals = [];
        for (let j = 0; j < tds.length; j++) {
            let txt = tds[j].textContent.trim();
            if (j === 0) {
                txt = txt.replace(/\s*Copy$/, "").trim();
            }
            vals.push(toCsvCell(txt));
        }
        lines.push(vals.join(","));
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "gaia_selected_rows.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

function writeInputs() {
    let start_row = document.createElement("tr");
    start_row.className = "input-row";
    let end_row = document.createElement("tr");
    end_row.className = "input-row";

    let start = document.createElement("td");
    start.className = "start-cell";
    start.innerHTML = "Start:";
    start_row.appendChild(start);

    let end = document.createElement("td");
    end.className = "start-cell";
    end.innerHTML = "End:";
    end_row.appendChild(end);

    for (let header in headers_array) {
        let start_cell = document.createElement("td");
        start_cell.className = "input-cell";
        let end_cell = document.createElement("td");
        end_cell.className = "input-cell";

        let input_start = document.createElement("input");
        let input_end = document.createElement("input");

        input_start.id = `${headers_array[header]}_start`;
        input_end.id = `${headers_array[header]}_end`;

        input_start.placeholder = `Start of ${headers_array[header]}`;
        input_end.placeholder = `End of ${headers_array[header]}`;

        start_cell.appendChild(input_start);
        end_cell.appendChild(input_end);

        start_row.appendChild(start_cell);
        end_row.appendChild(end_cell);

        input_start.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                searchFunction();
            }
        });
        input_start.addEventListener("input", function () {
            currentPage = 1;
            applyFiltersAndPagination();
        });
        input_end.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                searchFunction();
            }
        });
        input_end.addEventListener("input", function () {
            currentPage = 1;
            applyFiltersAndPagination();
        });
    }

    input_container.appendChild(start_row);
    input_container.appendChild(end_row);

    let submit_text = document.createElement("div");
    submit_text.innerHTML = "Submit query:";
    submit_text.style.display = "inline-block";
    input_box.appendChild(submit_text);

    let submit_button = document.createElement("button");
    submit_button.id = "submit-button";
    submit_button.innerHTML = "Submit";
    submit_button.addEventListener("click", function () { searchFunction(); });
    input_box.appendChild(submit_button);

    let reset_text = document.createElement("div");
    reset_text.innerHTML = "Reset all fields:";
    reset_text.style.display = "inline-block";
    input_box.appendChild(reset_text);

    let reset_button = document.createElement("button");
    reset_button.id = "reset-button";
    reset_button.innerHTML = "Reset";
    reset_button.addEventListener("click", function () { reset(); });
    input_box.appendChild(reset_button);

    const starInput = document.getElementById("STARinput");
    const toggleRv = document.getElementById("toggle-rv");
    const toggleFlux = document.getElementById("toggle-flux");
    const toggleSource = document.getElementById("toggle-source");
    const toggleApfData = document.getElementById("toggle-apf-data");
    const toggleKeck = document.getElementById("toggle-keck");
    const toggleRecentApf = document.getElementById("toggle-recent-apf");
    const toggleRecentKeck = document.getElementById("toggle-recent-keck");
    const toggleRecentApfWeek = document.getElementById("toggle-recent-apf-week");
    const toggleRecentKeckWeek = document.getElementById("toggle-recent-keck-week");
    const yesNoToggleControls = YES_NO_PARAMS.map(param => document.getElementById(param.toggleId));
    const pageSizeSelect = document.getElementById("page-size");
    const firstBtn = document.getElementById("page-first");
    const prevBtn = document.getElementById("page-prev");
    const nextBtn = document.getElementById("page-next");
    const lastBtn = document.getElementById("page-last");
    const copyViewLinkBtn = document.getElementById("copy-view-link");
    const downloadSelectedBtn = document.getElementById("download-selected");
    const sortApfCountBtn = document.getElementById("sort-apf-count");
    const sortKeckCountBtn = document.getElementById("sort-kpf-count");
    const sortApfRecentBtn = document.getElementById("sort-apf-recent");
    const sortKeckRecentBtn = document.getElementById("sort-kpf-recent");

    suppressUrlSync = true;
    if (starInput) {
        starInput.value = initialState.q;
    }
    if (toggleRv) toggleRv.checked = initialState.rv;
    if (toggleFlux) toggleFlux.checked = initialState.flux;
    if (toggleSource) toggleSource.checked = initialState.source;
    if (toggleApfData) toggleApfData.checked = initialState.apf;
    if (toggleKeck) toggleKeck.checked = initialState.keck;
    if (toggleRecentApf) toggleRecentApf.checked = initialState.rapf;
    if (toggleRecentKeck) toggleRecentKeck.checked = initialState.rkeck;
    if (toggleRecentApfWeek) toggleRecentApfWeek.checked = initialState.w7apf;
    if (toggleRecentKeckWeek) toggleRecentKeckWeek.checked = initialState.w7keck;
    for (let i = 0; i < YES_NO_PARAMS.length; i++) {
        if (yesNoToggleControls[i] && typeof yesNoToggleControls[i].value !== "undefined") {
            yesNoToggleControls[i].value = initialState[YES_NO_PARAMS[i].datasetKey] || "";
        }
    }
    if (pageSizeSelect && ["50", "100", "all"].includes(initialState.rows)) {
        pageSizeSelect.value = initialState.rows;
        pageSize = initialState.rows;
    }
    if (!Number.isNaN(initialState.page) && initialState.page > 0) {
        currentPage = initialState.page;
    }
    if (initialState.sort !== null && !Number.isNaN(parseInt(initialState.sort, 10))) {
        sortColumn = parseInt(initialState.sort, 10);
        sortDirection = initialState.dir === "desc" ? "desc" : "asc";
    }
    for (let i = 0; i < headers_array.length; i++) {
        const label = headers_array[i];
        const start = document.getElementById(`${label}_start`);
        const end = document.getElementById(`${label}_end`);
        if (start && end && initialState.ranges[label]) {
            start.value = initialState.ranges[label][0];
            end.value = initialState.ranges[label][1];
        }
    }
    suppressUrlSync = false;

    starInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            starSearch();
        }
    });
    starInput.addEventListener("input", function () {
        currentPage = 1;
        applyFiltersAndPagination();
    });

    if (toggleRv) toggleRv.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    if (toggleFlux) toggleFlux.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    if (toggleSource) toggleSource.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    if (toggleApfData) toggleApfData.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    if (toggleKeck) toggleKeck.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    if (toggleRecentApf) toggleRecentApf.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    if (toggleRecentKeck) toggleRecentKeck.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    if (toggleRecentApfWeek) toggleRecentApfWeek.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    if (toggleRecentKeckWeek) toggleRecentKeckWeek.addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
    for (let i = 0; i < yesNoToggleControls.length; i++) {
        if (yesNoToggleControls[i]) {
            yesNoToggleControls[i].addEventListener("change", () => { currentPage = 1; applyFiltersAndPagination(); });
        }
    }

    if (pageSizeSelect) {
        pageSizeSelect.addEventListener("change", function () {
            pageSize = this.value;
            currentPage = 1;
            applyFiltersAndPagination();
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener("click", function () {
            if (currentPage > 1) {
                currentPage -= 1;
                applyFiltersAndPagination();
            }
        });
    }

    if (firstBtn) {
        firstBtn.addEventListener("click", function () {
            currentPage = 1;
            applyFiltersAndPagination();
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener("click", function () {
            const rows = getDataRows();
            const matched = rows.filter(r => rowMatchesFilters(r));
            const perPage = pageSize === "all" ? matched.length : parseInt(pageSize, 10);
            const totalPages = Math.max(1, pageSize === "all" ? 1 : Math.ceil(matched.length / perPage));
            if (currentPage < totalPages) {
                currentPage += 1;
                applyFiltersAndPagination();
            }
        });
    }

    if (lastBtn) {
        lastBtn.addEventListener("click", function () {
            const rows = getDataRows();
            const matched = rows.filter(r => rowMatchesFilters(r));
            const perPage = pageSize === "all" ? matched.length : parseInt(pageSize, 10);
            const totalPages = Math.max(1, pageSize === "all" ? 1 : Math.ceil(matched.length / perPage));
            currentPage = totalPages;
            applyFiltersAndPagination();
        });
    }

    if (copyViewLinkBtn) {
        copyViewLinkBtn.addEventListener("click", function () {
            copyViewLink(copyViewLinkBtn);
        });
    }

    if (downloadSelectedBtn) {
        downloadSelectedBtn.addEventListener("click", function () {
            downloadSelectedCsv();
        });
    }

    if (sortApfCountBtn) {
        sortApfCountBtn.addEventListener("click", function () {
            sortRowsByApfCount();
        });
    }

    if (sortKeckCountBtn) {
        sortKeckCountBtn.addEventListener("click", function () {
            sortRowsByKeckCount();
        });
    }

    if (sortApfRecentBtn) {
        sortApfRecentBtn.addEventListener("click", function () {
            sortRowsByApfRecent();
        });
    }

    if (sortKeckRecentBtn) {
        sortKeckRecentBtn.addEventListener("click", function () {
            sortRowsByKeckRecent();
        });
    }
}

Promise.all([
    fetch("tables/data.csv")
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to load CSV: ${response.status}`);
            }
            considerLastUpdated(response.headers.get("last-modified"));
            return response.text();
        }),
    loadOptionalKeckCatalog(),
    loadOptionalSimbadCatalog(),
    refreshLastUpdatedFromWebsiteFiles()
])
    .then(([data]) => {
        writeInputs();
        loadCSV(data);
    })
    .catch(() => {
        input_container.innerHTML = "Failed to load table.";
        if (last_updated) {
            last_updated.innerHTML = "Last updated: unavailable";
        }
    });

table.addEventListener("click", async function (event) {
    const btn = event.target.closest(".copy-gaia-btn");
    if (!btn) {
        return;
    }
    event.preventDefault();
    let value = btn.getAttribute("data-copy") || "";
    const rowText = btn.parentElement ? btn.parentElement.textContent : "";
    const idMatch = rowText.match(/(\d{8,})/);
    if (idMatch && idMatch[1]) {
        value = idMatch[1];
    }
    value = String(value).replace(/\D/g, "");
    if (!value) {
        return;
    }
    try {
        await navigator.clipboard.writeText(value);
        const originalBg = btn.style.backgroundColor;
        const originalColor = btn.style.color;
        btn.style.backgroundColor = "#fff4b3";
        btn.style.color = "#111";
        setTimeout(() => {
            btn.style.backgroundColor = originalBg;
            btn.style.color = originalColor;
        }, 900);
    } catch {
        const originalBg = btn.style.backgroundColor;
        const originalColor = btn.style.color;
        btn.style.backgroundColor = "#b71c1c";
        btn.style.color = "#fff";
        setTimeout(() => {
            btn.style.backgroundColor = originalBg;
            btn.style.color = originalColor;
        }, 900);
    }
});
setInterval(renderLastUpdated, 60000);

document.addEventListener("keydown", function (event) {
    const tag = (event.target && event.target.tagName) ? event.target.tagName.toLowerCase() : "";
    const typing = tag === "input" || tag === "textarea";
    const starInput = document.getElementById("STARinput");

    if (event.key === "/" && !typing) {
        event.preventDefault();
        if (starInput) starInput.focus();
        return;
    }

    if (event.key === "Escape") {
        if (starInput) {
            starInput.value = "";
            currentPage = 1;
            applyFiltersAndPagination();
        }
        return;
    }

    if (typing) {
        return;
    }

    if (event.key.toLowerCase() === "n") {
        const nextBtn = document.getElementById("page-next");
        if (nextBtn) nextBtn.click();
        return;
    }
    if (event.key.toLowerCase() === "p") {
        const prevBtn = document.getElementById("page-prev");
        if (prevBtn) prevBtn.click();
    }
});
