const form = document.getElementById("analyze-form");
const tickerInput = document.getElementById("ticker-input");
const periodSelect = document.getElementById("period-select");
const statusPill = document.getElementById("status-pill");

const metricPrice = document.getElementById("metric-price");
const metricMarketCap = document.getElementById("metric-marketcap");
const metricPe = document.getElementById("metric-pe");
const metricAth = document.getElementById("metric-ath");

const swingReco = document.getElementById("swing-reco");
const swingConfidence = document.getElementById("swing-confidence");
const basicReco = document.getElementById("basic-reco");
const basicConfidence = document.getElementById("basic-confidence");
const sentimentScore = document.getElementById("sentiment-score");
const articleCount = document.getElementById("article-count");
const companyLabel = document.getElementById("company-label");

const alertsList = document.getElementById("alerts-list");
const newsList = document.getElementById("news-list");
const basicReason = document.getElementById("basic-reason");

const priceChartId = "price-chart";
const rsiChartId = "rsi-chart";
const macdChartId = "macd-chart";

function formatCurrency(value, currency = "USD") {
  if (value === null || value === undefined) return "N/A";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(value);
  } catch {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 2,
    }).format(value);
  }
}

function formatCompactCurrency(value, currency = "USD") {
  if (value === null || value === undefined) return "N/A";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      notation: "compact",
      maximumFractionDigits: 2,
    }).format(value);
  } catch {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      notation: "compact",
      maximumFractionDigits: 2,
    }).format(value);
  }
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined) return "N/A";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value) {
  if (value === null || value === undefined) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

function normalizeTone(text) {
  if (!text) return "tone-hold";
  const upper = String(text).toUpperCase();
  if (upper.includes("BUY")) return "tone-buy";
  if (upper.includes("SELL")) return "tone-sell";
  return "tone-hold";
}

function setStatus(message, tone) {
  statusPill.textContent = message;
  statusPill.className = `status-pill status-${tone}`;
}

function listHasValues(values) {
  return Array.isArray(values) && values.some((v) => v !== null && v !== undefined);
}

function drawPriceChart(data) {
  const traces = [
    {
      x: data.dates,
      open: data.open,
      high: data.high,
      low: data.low,
      close: data.close,
      type: "candlestick",
      name: "Price",
      xaxis: "x",
      yaxis: "y",
      increasing: { line: { color: "#2ecf9f" } },
      decreasing: { line: { color: "#ff7f78" } },
    },
    {
      x: data.dates,
      y: data.volume,
      type: "bar",
      name: "Volume",
      xaxis: "x",
      yaxis: "y2",
      marker: { color: "rgba(137, 168, 198, 0.34)" },
    },
  ];

  if (listHasValues(data.sma20)) {
    traces.push({
      x: data.dates,
      y: data.sma20,
      type: "scatter",
      mode: "lines",
      name: "SMA 20",
      line: { width: 1.2, color: "#f8bf66" },
      xaxis: "x",
      yaxis: "y",
    });
  }

  if (listHasValues(data.sma50)) {
    traces.push({
      x: data.dates,
      y: data.sma50,
      type: "scatter",
      mode: "lines",
      name: "SMA 50",
      line: { width: 1.3, color: "#63dbc7" },
      xaxis: "x",
      yaxis: "y",
    });
  }

  if (listHasValues(data.sma200)) {
    traces.push({
      x: data.dates,
      y: data.sma200,
      type: "scatter",
      mode: "lines",
      name: "SMA 200",
      line: { width: 1.6, color: "#5ea8ff" },
      xaxis: "x",
      yaxis: "y",
    });
  }

  const layout = {
    margin: { l: 44, r: 16, t: 12, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#cde2f8", family: "Space Grotesk, sans-serif" },
    legend: { orientation: "h", y: 1.08 },
    xaxis: {
      showgrid: true,
      gridcolor: "rgba(121,156,191,0.15)",
      rangeslider: { visible: false },
    },
    yaxis: {
      domain: [0.31, 1],
      title: "Price",
      showgrid: true,
      gridcolor: "rgba(121,156,191,0.15)",
    },
    yaxis2: {
      domain: [0, 0.22],
      title: "Volume",
      showgrid: true,
      gridcolor: "rgba(121,156,191,0.12)",
    },
  };

  Plotly.newPlot(priceChartId, traces, layout, { responsive: true, displayModeBar: false });
}

function drawRsiChart(data) {
  if (!listHasValues(data.rsi14)) {
    Plotly.purge(rsiChartId);
    return;
  }

  const traces = [
    {
      x: data.dates,
      y: data.rsi14,
      type: "scatter",
      mode: "lines",
      name: "RSI 14",
      line: { color: "#5ea8ff", width: 2 },
    },
  ];

  const layout = {
    margin: { l: 42, r: 16, t: 12, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#cde2f8", family: "Space Grotesk, sans-serif" },
    showlegend: false,
    xaxis: {
      showgrid: true,
      gridcolor: "rgba(121,156,191,0.12)",
    },
    yaxis: {
      range: [0, 100],
      showgrid: true,
      gridcolor: "rgba(121,156,191,0.12)",
    },
    shapes: [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        y0: 70,
        y1: 70,
        line: { color: "#ff7f78", dash: "dash" },
      },
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        y0: 30,
        y1: 30,
        line: { color: "#2ecf9f", dash: "dash" },
      },
    ],
  };

  Plotly.newPlot(rsiChartId, traces, layout, { responsive: true, displayModeBar: false });
}

function drawMacdChart(data) {
  if (!listHasValues(data.macd) || !listHasValues(data.macd_signal)) {
    Plotly.purge(macdChartId);
    return;
  }

  const histColors = (data.macd_hist || []).map((value) =>
    value === null || value === undefined ? "rgba(121,156,191,0.15)" : value >= 0 ? "#2ecf9f" : "#ff7f78"
  );

  const traces = [
    {
      x: data.dates,
      y: data.macd,
      type: "scatter",
      mode: "lines",
      name: "MACD",
      line: { color: "#5ea8ff", width: 2 },
    },
    {
      x: data.dates,
      y: data.macd_signal,
      type: "scatter",
      mode: "lines",
      name: "Signal",
      line: { color: "#f8bf66", width: 1.8 },
    },
  ];

  if (listHasValues(data.macd_hist)) {
    traces.push({
      x: data.dates,
      y: data.macd_hist,
      type: "bar",
      name: "Hist",
      marker: { color: histColors },
      opacity: 0.75,
    });
  }

  const layout = {
    margin: { l: 42, r: 16, t: 12, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#cde2f8", family: "Space Grotesk, sans-serif" },
    xaxis: { showgrid: true, gridcolor: "rgba(121,156,191,0.12)" },
    yaxis: { showgrid: true, gridcolor: "rgba(121,156,191,0.12)" },
    legend: { orientation: "h", y: 1.08 },
  };

  Plotly.newPlot(macdChartId, traces, layout, { responsive: true, displayModeBar: false });
}

function renderList(el, items, fallback) {
  el.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = fallback;
    el.appendChild(li);
    return;
  }

  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    el.appendChild(li);
  });
}

function applyPayload(payload) {
  const currency = payload.quote?.currency || "USD";
  const swing = payload.analysis?.swing_trader || {};
  const basic = payload.analysis?.basic || {};
  const sentiment = payload.news?.overall_sentiment;

  metricPrice.textContent = formatCurrency(payload.quote?.current_price, currency);
  metricMarketCap.textContent = formatCompactCurrency(payload.fundamentals?.market_cap, currency);
  metricPe.textContent = formatNumber(payload.fundamentals?.trailing_pe);
  metricAth.textContent = formatCurrency(payload.fundamentals?.ath_from_period, currency);

  swingReco.textContent = swing.recommendation || "N/A";
  swingReco.className = normalizeTone(swing.recommendation);
  swingConfidence.textContent = `Confidence: ${formatNumber(swing.confidence, 0)}%`;

  basicReco.textContent = basic.recommendation || "N/A";
  basicReco.className = normalizeTone(basic.recommendation);
  basicConfidence.textContent = `Confidence: ${formatNumber(basic.confidence, 0)}%`;

  sentimentScore.textContent =
    sentiment === null || sentiment === undefined ? "N/A" : formatNumber(sentiment, 2);
  sentimentScore.className = sentiment > 0.1 ? "tone-buy" : sentiment < 0 ? "tone-sell" : "tone-hold";
  articleCount.textContent = `Articles: ${payload.news?.article_count ?? 0}`;

  companyLabel.textContent = `${payload.meta?.company_name || payload.meta?.ticker || "Ticker"} (${payload.meta?.period || "-"})`;
  basicReason.textContent = basic.reason || "No reason available.";

  renderList(alertsList, swing.alerts || [], "No active swing alerts for this run.");
  renderList(newsList, payload.news?.titles?.slice(0, 12) || [], "No relevant news headlines were found.");

  drawPriceChart(payload.charts);
  drawRsiChart(payload.charts);
  drawMacdChart(payload.charts);
}

async function analyzeStock() {
  const ticker = tickerInput.value.trim().toUpperCase();
  const period = periodSelect.value;

  if (!ticker) {
    setStatus("Ticker is required.", "error");
    return;
  }

  setStatus(`Analyzing ${ticker}...`, "loading");

  try {
    const response = await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(period)}`);
    const payload = await response.json();

    if (!response.ok || payload.status !== "ok") {
      throw new Error(payload.error || "Analysis failed.");
    }

    applyPayload(payload);
    setStatus(`Updated ${ticker} successfully.`, "success");
  } catch (error) {
    setStatus(error.message || "Request failed.", "error");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  analyzeStock();
});

analyzeStock();
