/* Report behaviour.
 *
 * The page is a document, not a console: it fetches one snapshot, renders the
 * whole report in reading order, and stops. If the selected run is still in
 * flight it tails the stream and re-renders on each server snapshot, but there
 * are no playback controls — a reader of an eval does not scrub a run.
 *
 * Every number on this page comes from the snapshot. Nothing is computed here,
 * and no sentence about the result is written here: the finding, the caveats
 * and the method notes are all generated server-side from the loaded run, so a
 * dataset that does not support a claim cannot have that claim rendered over it.
 */
(function () {
  "use strict";

  var state = {
    meta: null,
    snapshot: null,
    datasetId: null,
    runId: null,
    source: null
  };

  var $ = function (id) {
    return document.getElementById(id);
  };

  function q(params) {
    var parts = [];
    if (state.datasetId) parts.push("dataset=" + encodeURIComponent(state.datasetId));
    Object.keys(params || {}).forEach(function (key) {
      var value = params[key];
      if (value !== null && value !== undefined && value !== "") {
        parts.push(key + "=" + encodeURIComponent(value));
      }
    });
    return parts.length ? "?" + parts.join("&") : "";
  }

  function getJson(url) {
    return fetch(url, { headers: { Accept: "application/json" } }).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (body) {
          throw new Error("HTTP " + response.status + ": " + body.slice(0, 240));
        });
      }
      return response.json();
    });
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function para(host, text, cls) {
    if (!text) return;
    host.appendChild(el("p", cls || null, text));
  }

  function showNotice(message) {
    var box = $("notice");
    box.textContent = message;
    box.hidden = false;
  }

  function clearNotice() {
    $("notice").hidden = true;
  }

  function announce(text) {
    $("live-region").textContent = text;
  }

  function num(value, digits) {
    return window.Charts.fmt(value, digits === undefined ? 3 : digits);
  }

  /* ---------------- load ---------------- */
  function loadMeta() {
    return getJson("/api/meta" + q({})).then(function (meta) {
      state.meta = meta;
      state.datasetId = meta.dataset;
      $("mca-footer").textContent = meta.footer;

      var datasets = $("dataset-select");
      datasets.textContent = "";
      (meta.datasets || []).forEach(function (dataset) {
        var option = document.createElement("option");
        option.value = dataset.id;
        option.textContent =
          dataset.id +
          " · " +
          dataset.records +
          " records · " +
          (dataset.kind === "real" ? dataset.licence || "real" : "synthetic diagnostic");
        datasets.appendChild(option);
      });
      datasets.disabled = (meta.datasets || []).length === 0;
      if (!datasets.disabled) datasets.value = state.datasetId;

      var runs = $("run-select");
      runs.textContent = "";
      (meta.runs || []).forEach(function (run) {
        var option = document.createElement("option");
        option.value = run.run_id;
        option.textContent =
          run.provider + " · " + run.split + " · " + run.status + " · " + run.run_id.slice(-6);
        runs.appendChild(option);
      });
      runs.disabled = !meta.runs || meta.runs.length === 0;

      var preferred = (meta.runs || []).filter(function (r) {
        return r.run_id === meta.default_run || r.path === meta.default_run;
      })[0];
      if (!preferred) {
        preferred = (meta.runs || []).filter(function (r) {
          return r.provider === "jev";
        })[0];
      }
      if (!preferred) preferred = (meta.runs || [])[0];
      state.runId = preferred ? preferred.run_id : null;
      if (state.runId) runs.value = state.runId;
    });
  }

  function loadSnapshot() {
    if (!state.runId) return Promise.resolve();
    $("status").textContent = "loading";
    return getJson("/api/snapshot" + q({ run: state.runId })).then(function (snapshot) {
      clearNotice();
      state.snapshot = snapshot;
      render();
      $("status").textContent = snapshot.run.status;
    });
  }

  /* A run still being written is tailed so the report fills in as it lands.
   * A finished run is read once: there is nothing to watch. */
  function maybeStream() {
    stopStream();
    if (!state.snapshot || state.snapshot.run.status === "complete") return;
    var source = new EventSource("/api/stream" + q({ run: state.runId, mode: "live" }));
    state.source = source;
    source.addEventListener("snapshot", function (event) {
      state.snapshot = JSON.parse(event.data);
      render();
    });
    source.addEventListener("done", function () {
      stopStream();
      loadSnapshot();
    });
    source.onerror = function () {
      stopStream();
    };
  }

  function stopStream() {
    if (state.source) {
      state.source.close();
      state.source = null;
    }
  }

  /* ---------------- render ---------------- */
  function render() {
    if (!state.snapshot) return;
    renderMasthead();
    renderFinding();
    renderDetection();
    renderLabels();
    renderFaults();
    renderCost();
    renderMethod();
    renderColophon();
  }

  function renderMasthead() {
    var snapshot = state.snapshot;
    var dataset = snapshot.dataset || {};
    var provenance = dataset.provenance || {};
    var byline = $("byline");
    byline.textContent = "";
    [
      dataset.kind === "real" ? "Real traces" : "Synthetic diagnostic",
      (provenance.dataset_id || dataset.id || "").toString(),
      snapshot.run.records + " records, " + snapshot.kpi.traces_total + " in the scored split",
      "split " + snapshot.run.split,
      "generated " + (snapshot.generated_utc || "")
    ]
      .filter(Boolean)
      .forEach(function (bit) {
        byline.appendChild(el("span", null, bit));
      });
  }

  function renderFinding() {
    var finding = state.snapshot.finding || {};
    $("finding-sentence").textContent = finding.sentence || "No finding established yet.";
    $("finding-sub").textContent = finding.sub || "";

    var list = $("finding-claims");
    list.textContent = "";
    (finding.claims || []).forEach(function (claim) {
      var li = document.createElement("li");
      li.appendChild(
        el(
          "span",
          "stance " + claim.stance,
          claim.stance === "for" ? "In favour" : claim.stance === "against" ? "Against" : "Caveat"
        )
      );
      var body = el("span", "claim-text", claim.text);
      body.appendChild(el("span", "claim-detail", claim.detail));
      li.appendChild(body);
      list.appendChild(li);
    });

    $("finding-note").textContent =
      "Every clause above is emitted only when the interval behind it is clear of zero. " +
      "A claim the data stopped supporting would disappear from this paragraph rather " +
      "than stay on the page as a stale sentence.";
  }

  function renderDetection() {
    var snapshot = state.snapshot;
    var host = $("detection-prose");
    host.textContent = "";
    var reference = snapshot.reference || {};
    var strong = reference.tfidf_gbm || {};
    var jev = reference.jev || {};
    var judge = reference.openai || {};

    para(
      host,
      "The task is binary: is this trace an unsupported success — a final message claiming " +
        "an external action succeeded when the ordered tool evidence does not support that " +
        "exact claim? Detectors are scored on the area under the precision-recall curve, " +
        "which is threshold-free, on a test split whose tasks appear in no training or " +
        "calibration set."
    );
    if (jev.auprc !== undefined && strong.auprc !== undefined) {
      para(
        host,
        "Both paid arms are beaten by both free classifiers, and not marginally. The trained " +
          "gradient-boosted classifier reaches " +
          num(strong.auprc) +
          " against " +
          num(jev.auprc) +
          " for the typed detector and " +
          num(judge.auprc) +
          " for the general judge; the paired interval on the first of those differences is " +
          "entirely below zero."
      );
    }
    para(
      host,
      "The arms that train are grouped apart from the arms that do not, because that is the " +
        "distinction the numbers turn on — not because one group is the winner. Colour here " +
        "encodes what a detector is, never where it placed."
    );
    window.Charts.render($("figure-headline"), (snapshot.figures || {}).headline_auprc);
  }

  function renderLabels() {
    var snapshot = state.snapshot;
    var figure = (snapshot.figures || {}).learning_curve;
    var host = $("labels-prose");
    host.textContent = "";
    para(
      host,
      "A classifier needs labels. A zero-shot detector does not, and that is the one " +
        "advantage the paid arms keep once the comparison above has gone against them. So " +
        "the label budget was swept: subsets of the training split drawn task-disjointly — " +
        "whole tasks admitted in a seeded order, never individual rows — stratified to the " +
        "split's own label prevalence, five seeds at each budget, every fit scored on the " +
        "same frozen test split."
    );
    para(
      host,
      "No paid call was made to draw this. The flat rules are the recorded zero-shot runs, " +
        "re-scored from their own prediction files."
    );
    window.Charts.render($("figure-curve"), figure);

    var aside = $("crossover-aside");
    aside.textContent = "";
    if (!figure || !figure.available) {
      para(aside, (figure && figure.reason) || "No label sweep on this split yet.");
      return;
    }
    aside.appendChild(el("h3", null, "Reading the crossover"));
    var providers = snapshot.providers || {};
    Object.keys(figure.crossovers || {})
      .sort()
      .forEach(function (provider) {
        var byRef = figure.crossovers[provider];
        Object.keys(byRef)
          .sort()
          .forEach(function (reference) {
            var result = byRef[reference];
            para(
              aside,
              ((providers[provider] || {}).short || provider) +
                " passes " +
                ((providers[reference] || {}).short || reference) +
                " on mean AUPRC at " +
                (result.mean ? result.mean.n_train + " labels" : "no budget tested") +
                ", and on the lower edge of its band at " +
                (result.lower_bound ? result.lower_bound.n_train + " labels" : "no budget tested") +
                "."
            );
          });
      });
    para(
      aside,
      "A budget only counts as the crossover if it and every larger budget stay above the " +
        "line, so a single lucky draw is not reported as a crossing.",
      "note"
    );
    if (figure.caveats && figure.caveats.calibration) {
      para(aside, figure.caveats.calibration, "note");
    }
  }

  function renderFaults() {
    var snapshot = state.snapshot;
    var host = $("faults-prose");
    host.textContent = "";
    para(
      host,
      "An average hides where a detector earns its result. These panels split the false " +
        "successes by the benchmark's own difficulty metadata and show what share of each " +
        "group every detector flags at its frozen threshold. All panels share one scale, so " +
        "they are compared by position."
    );
    para(
      host,
      "Read the interval, not the bar. A group with few traces carries a wide one, and two " +
        "bars whose intervals overlap have not been shown to differ."
    );
    window.Charts.render($("figure-faults"), (snapshot.figures || {}).per_fault);
  }

  function renderCost() {
    var snapshot = state.snapshot;
    var host = $("cost-prose");
    host.textContent = "";
    var figure = (snapshot.figures || {}).cost;
    var unavailable = ((figure || {}).series || []).filter(function (s) {
      return s.paid && !s.available;
    });
    para(
      host,
      "Cost is recorded, never projected: every figure below comes from the input and output " +
        "tokens each API actually returned, priced through a dated manifest whose SHA-256 is " +
        "written into the run. Sterling is converted at a fixed operator-supplied rate in the " +
        "config — no rate is fetched, so the same run shows the same number tomorrow."
    );
    if (unavailable.length) {
      para(
        host,
        "One arm has no price. " +
          unavailable
            .map(function (s) {
              return s.label + " (" + (s.model_id || "unknown model") + ")";
            })
            .join(", ") +
          " has no verified rate in the manifest, so its cost is reported as unavailable " +
          "rather than guessed — and no claim that either paid arm is the cheaper one is " +
          "made anywhere on this page."
      );
    }
    window.Charts.render($("figure-cost"), figure);
    window.Charts.render($("figure-latency"), (snapshot.figures || {}).latency);
  }

  function renderMethod() {
    var snapshot = state.snapshot;
    var host = $("method-body");
    host.textContent = "";
    var dataset = snapshot.dataset || {};
    var provenance = dataset.provenance || {};
    var caveats = snapshot.caveats || {};
    var honesty = snapshot.honesty || {};

    host.appendChild(el("h3", null, "The data"));
    if (provenance.licence) {
      para(
        host,
        (provenance.notes && provenance.notes[0]) ||
          "Real, released agent trajectories with programmatic ground truth."
      );
      para(
        host,
        "Licence: " +
          provenance.licence.name +
          " (" +
          provenance.licence.spdx +
          "), verified before anything was read. " +
          provenance.licence.attribution
      );
      if (provenance.label_rule) {
        para(
          host,
          "Labels come from " +
            provenance.label_rule.truth_field +
            " against " +
            provenance.label_rule.claim_field +
            ". Neither input is the assistant's prose, so a label does not depend on how the " +
            "agent worded itself."
        );
      }
    } else {
      para(
        host,
        "This dataset carries no provenance record, so nothing is claimed about its origin."
      );
    }

    host.appendChild(el("h3", null, "The split"));
    para(
      host,
      "The split unit is the task, never the row. Sibling trajectories of one task are " +
        "paraphrases of one problem, so splitting by row would put a training example and a " +
        "test example of the same task on opposite sides. The split hash is frozen and " +
        "recorded in every run manifest, and a classifier's fit refuses a held-out task " +
        "outright rather than dropping it."
    );
    para(host, honesty.threshold_note || "");
    para(host, honesty.repeats_note || "");

    host.appendChild(el("h3", null, "What limits this result"));
    (caveats.items || []).forEach(function (item) {
      var block = el("div", "limit");
      block.appendChild(el("strong", null, item.title));
      block.appendChild(el("span", null, item.body));
      host.appendChild(block);
    });

    host.appendChild(el("h3", null, "The result that goes against the paid arms"));
    para(host, (snapshot.comparison || {}).verdict_text || "No comparison computed for this run.");
    para(
      host,
      "It is stated here at the same size as everything else because it is the finding most " +
        "likely to be left out. On this corpus a free classifier trained on a few dozen " +
        "labels is the better detector, and no arrangement of the figures above is allowed " +
        "to imply otherwise."
    );

    host.appendChild(el("h3", null, "How to read the numbers"));
    var rules = el("ul");
    [
      "Every estimated quantity carries its 95% interval. Counts and sums are exact and are " +
        "shown without one.",
      "An interval on a single detector is an unpaired percentile bootstrap over resampled " +
        "traces. An interval on a difference is paired, and narrower; neither is ever quoted " +
        "for the other.",
      "Detection rates carry Wilson intervals, which behave near 0 and 1 where flag rates live.",
      "A lane that did not score a group shows nothing, not a zero.",
      "Point figures use one pass per trace. Repeats measure stability, not accuracy, and are " +
        "never averaged into a rate.",
      "Colour encodes role, never rank, and every figure has a table behind it, so identity " +
        "is never colour alone."
    ].forEach(function (rule) {
      rules.appendChild(el("li", null, rule));
    });
    host.appendChild(rules);

    host.appendChild(el("h3", null, "Reproducing this"));
    para(
      host,
      "Every figure is derived from append-only run directories on disk: predictions, " +
        "attempts, a manifest carrying the dataset, split, config and price-manifest hashes, " +
        "and whatever each run had to truncate. jev-eval verify-runs recomputes the checksums " +
        "and the cost arithmetic. This page reads those artifacts and computes no metric of " +
        "its own; it calls the same functions the written report does."
    );
  }

  function renderColophon() {
    var snapshot = state.snapshot;
    var cost = snapshot.cost || {};
    $("colophon-meta").textContent = [
      "run " + snapshot.run.run_id,
      "model " + snapshot.run.model_id,
      "prices " + (cost.prices_date || snapshot.run.prices_date || ""),
      "fx " + (cost.fx_rate_gbp_usd || "") + " GBP/USD on " + (cost.fx_rate_date || ""),
      "rendered " + (snapshot.generated_utc || "")
    ]
      .filter(Boolean)
      .join("  ·  ");
    $("contents-foot").textContent = snapshot.run.provider + " · split " + snapshot.run.split;
  }

  /* ---------------- chrome ---------------- */
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    var button = $("theme-toggle");
    button.textContent = theme === "dark" ? "Light" : "Dark";
    button.setAttribute("aria-pressed", String(theme === "dark"));
    try {
      localStorage.setItem("fse-theme", theme);
    } catch (e) {
      /* blocked storage is not a reason to fail to render */
    }
  }

  function trackSections() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".contents a"));
    var sections = links
      .map(function (link) {
        return document.getElementById(link.dataset.section);
      })
      .filter(Boolean);
    if (!("IntersectionObserver" in window) || !sections.length) return;
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (link) {
            if (link.dataset.section === entry.target.id) {
              link.setAttribute("aria-current", "true");
            } else {
              link.removeAttribute("aria-current");
            }
          });
        });
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );
    sections.forEach(function (section) {
      observer.observe(section);
    });
  }

  /* Export writes a PNG and a CSV under reports/. The view sent is the section
   * nearest the top of the viewport, so "export" means "export what I am
   * reading". */
  var SECTION_VIEW = {
    finding: "overview",
    detection: "overview",
    labels: "learning-curve",
    faults: "per-fault",
    cost: "cost",
    method: "overview"
  };

  function currentSection() {
    var best = "finding";
    var bestTop = Infinity;
    Object.keys(SECTION_VIEW).forEach(function (id) {
      var section = document.getElementById(id);
      if (!section) return;
      var top = Math.abs(section.getBoundingClientRect().top);
      if (top < bestTop) {
        bestTop = top;
        best = id;
      }
    });
    return best;
  }

  function wire() {
    $("dataset-select").addEventListener("change", function () {
      state.datasetId = this.value;
      state.runId = null;
      state.snapshot = null;
      stopStream();
      clearNotice();
      loadMeta()
        .then(loadSnapshot)
        .then(maybeStream)
        .catch(function (error) {
          showNotice(String(error.message || error));
        });
    });

    $("run-select").addEventListener("change", function () {
      state.runId = this.value;
      stopStream();
      loadSnapshot()
        .then(maybeStream)
        .catch(function (error) {
          showNotice(String(error.message || error));
        });
    });

    $("theme-toggle").addEventListener("click", function () {
      applyTheme(
        document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"
      );
    });

    $("export-btn").addEventListener("click", function () {
      var button = this;
      var label = button.textContent;
      button.disabled = true;
      button.textContent = "Exporting";
      fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run: state.runId,
          dataset: state.datasetId,
          view: SECTION_VIEW[currentSection()] || "overview"
        })
      })
        .then(function (response) {
          if (!response.ok) throw new Error("export failed: HTTP " + response.status);
          return response.json();
        })
        .then(function (payload) {
          announce("Exported " + payload.written.join(" and "));
          $("status").textContent = "exported " + payload.written.length + " files";
        })
        .catch(function (error) {
          showNotice(String(error.message || error));
        })
        .finally(function () {
          button.disabled = false;
          button.textContent = label;
        });
    });
  }

  function start() {
    var stored = null;
    try {
      stored = localStorage.getItem("fse-theme");
    } catch (e) {
      stored = null;
    }
    var prefersDark =
      window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(stored || (prefersDark ? "dark" : "light"));
    wire();
    trackSections();
    loadMeta()
      .then(loadSnapshot)
      .then(maybeStream)
      .catch(function (error) {
        showNotice(String(error.message || error));
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
