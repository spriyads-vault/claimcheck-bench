/* The whole charting layer. Inline SVG, no dependency, no build step, nothing
 * fetched — the harness is offline-first and the page must stay that way.
 *
 * Three rules are enforced here rather than left to the caller:
 *
 *   1. Colour carries role, never rank. A series' hue comes from its `role`
 *      field, which the server sets from what the detector *is* (trained,
 *      typed, general, heuristic), not from where it placed.
 *   2. Values and labels wear text colours. `var(--text)` and `var(--muted)`
 *      are the only fills used for type. A number never takes its series' hue,
 *      so a reader who cannot separate two hues still reads every figure.
 *   3. Every chart takes the same figure spec the table view renders from, so
 *      the picture and the numbers behind it cannot drift.
 */
(function (global) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  /* Role -> CSS custom property. The only place a series colour is decided. */
  var ROLE_VAR = {
    trained: "--series-trained",
    "trained-alt": "--series-trained-alt",
    typed: "--series-typed",
    general: "--series-general",
    heuristic: "--series-heuristic"
  };

  function roleColour(role) {
    return "var(" + (ROLE_VAR[role] || "--series-heuristic") + ")";
  }

  function node(tag, attrs) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) {
      n.setAttribute(k, attrs[k]);
    });
    return n;
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function fmt(value, digits) {
    return isNum(value) ? value.toFixed(digits === undefined ? 3 : digits) : "—";
  }

  /* ---- tooltip ---------------------------------------------------------
   * One element for the whole page, moved rather than recreated. Bound to
   * pointer AND focus, so a keyboard reader gets the same detail; the table
   * behind every figure is the third route to the same numbers. */
  var tip = null;

  function tooltip() {
    if (!tip) tip = document.getElementById("tooltip");
    return tip;
  }

  function showTip(event, title, lines) {
    var t = tooltip();
    if (!t) return;
    t.textContent = "";
    t.appendChild(el("span", "tip-title", title));
    (lines || []).forEach(function (line) {
      t.appendChild(el("span", "tip-value", line));
      t.appendChild(document.createElement("br"));
    });
    var box = event.target.getBoundingClientRect();
    t.style.left = box.left + box.width / 2 + "px";
    t.style.top = box.top - 8 + "px";
    t.setAttribute("data-visible", "true");
    t.setAttribute("aria-hidden", "false");
  }

  function hideTip() {
    var t = tooltip();
    if (!t) return;
    t.setAttribute("data-visible", "false");
    t.setAttribute("aria-hidden", "true");
  }

  function bindTip(target, title, lines) {
    target.addEventListener("mouseenter", function (e) {
      showTip(e, title, lines);
    });
    target.addEventListener("mouseleave", hideTip);
    target.addEventListener("focus", function (e) {
      showTip(e, title, lines);
    });
    target.addEventListener("blur", hideTip);
    target.setAttribute("tabindex", "0");
    target.setAttribute("role", "img");
    target.setAttribute("aria-label", title + ". " + (lines || []).join(". "));
  }

  function svgRoot(width, height, label) {
    var svg = node("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": label || ""
    });
    svg.setAttribute("width", "100%");
    return svg;
  }

  function axisText(x, y, text, anchor, size) {
    var t = node("text", {
      x: x,
      y: y,
      "text-anchor": anchor || "middle",
      "font-size": size || 11,
      fill: "var(--muted)"
    });
    t.textContent = text;
    return t;
  }

  /* ---- interval dots ---------------------------------------------------
   * A point estimate and its 95% interval, one row per series, grouped by the
   * `group` field. Horizontal because the labels are words, and a horizontal
   * row lets each one be direct-labelled instead of rotated under an axis. */
  function intervalDots(container, figure) {
    container.textContent = "";
    var series = (figure.series || []).filter(function (s) {
      return s.points && s.points.length;
    });
    if (!series.length) {
      container.appendChild(el("div", "empty", "No scored runs on this split yet."));
      return;
    }

    var rowH = 46;
    var padTop = 16;
    var padBottom = 42;
    var padLeft = 176;
    var padRight = 28;
    var groups = [];
    series.forEach(function (s) {
      if (!groups.length || groups[groups.length - 1].name !== s.group) {
        groups.push({ name: s.group, rows: [] });
      }
      groups[groups.length - 1].rows.push(s);
    });
    var groupHeadH = 26;
    var height =
      padTop + padBottom + series.length * rowH + groups.length * groupHeadH;
    var width = 720;

    var values = [];
    series.forEach(function (s) {
      s.points.forEach(function (p) {
        [p.y, p.lower, p.upper].forEach(function (v) {
          if (isNum(v)) values.push(v);
        });
      });
    });
    var domain = figure.domain;
    var lo = domain ? domain[0] : Math.min.apply(null, values);
    var hi = domain ? domain[1] : Math.max.apply(null, values);
    if (!domain) {
      var pad = (hi - lo) * 0.18 || Math.max(0.01, hi * 0.2);
      lo = Math.max(0, lo - pad);
      hi = hi + pad;
    }
    if (hi <= lo) hi = lo + 1;

    function px(v) {
      return padLeft + ((v - lo) / (hi - lo)) * (width - padLeft - padRight);
    }

    var svg = svgRoot(width, height, figure.title);
    var ticks = 5;
    for (var i = 0; i <= ticks; i++) {
      var v = lo + ((hi - lo) * i) / ticks;
      svg.appendChild(
        node("line", {
          x1: px(v), x2: px(v), y1: padTop, y2: height - padBottom,
          stroke: "var(--grid)", "stroke-width": 1
        })
      );
      svg.appendChild(axisText(px(v), height - padBottom + 18, fmt(v, figure.digits > 3 ? 3 : 2)));
    }
    svg.appendChild(
      axisText((padLeft + width - padRight) / 2, height - padBottom + 36, figure.value_label)
    );

    var y = padTop;
    groups.forEach(function (group) {
      var head = node("text", {
        x: 0, y: y + 14, "font-size": 10.5, fill: "var(--muted)",
        "letter-spacing": "0.08em"
      });
      head.textContent = group.name.toUpperCase();
      svg.appendChild(head);
      svg.appendChild(
        node("line", {
          x1: 0, x2: width - padRight, y1: y + 21, y2: y + 21,
          stroke: "var(--line)", "stroke-width": 1
        })
      );
      y += groupHeadH;

      group.rows.forEach(function (s) {
        var point = s.points[0];
        var cy = y + rowH / 2;
        var colour = roleColour(s.role);

        var name = node("text", {
          x: 0, y: cy - 3, "font-size": 13, fill: "var(--text)"
        });
        name.textContent = s.label;
        svg.appendChild(name);
        var sub = node("text", {
          x: 0, y: cy + 13, "font-size": 10.5, fill: "var(--muted)"
        });
        sub.textContent = s.paid ? "paid" : s.trains ? "free, trained" : "free";
        svg.appendChild(sub);

        if (!isNum(point.y)) {
          var missing = node("text", {
            x: padLeft, y: cy + 4, "font-size": 12, fill: "var(--muted)"
          });
          missing.textContent = s.unavailable_reason || "not available";
          svg.appendChild(missing);
          y += rowH;
          return;
        }

        if (isNum(point.lower) && isNum(point.upper) && point.upper > point.lower) {
          svg.appendChild(
            node("line", {
              x1: px(point.lower), x2: px(point.upper), y1: cy, y2: cy,
              stroke: colour, "stroke-width": 2, "stroke-linecap": "round", opacity: 0.55
            })
          );
          [point.lower, point.upper].forEach(function (edge) {
            svg.appendChild(
              node("line", {
                x1: px(edge), x2: px(edge), y1: cy - 5, y2: cy + 5,
                stroke: colour, "stroke-width": 2, "stroke-linecap": "round"
              })
            );
          });
        }
        var dot = node("circle", { cx: px(point.y), cy: cy, r: 5, fill: colour });
        svg.appendChild(dot);
        var lines = [figure.value_label + " " + fmt(point.y, figure.digits)];
        if (isNum(point.lower)) {
          lines.push("95% CI " + fmt(point.lower, figure.digits) + " to " + fmt(point.upper, figure.digits));
        }
        Object.keys(point.extra || {}).forEach(function (k) {
          lines.push(k + " " + point.extra[k]);
        });
        bindTip(dot, s.label, lines);

        var value = node("text", {
          x: px(point.y), y: cy - 12, "text-anchor": "middle",
          "font-size": 12, fill: "var(--text)"
        });
        value.textContent = fmt(point.y, figure.digits);
        svg.appendChild(value);

        y += rowH;
      });
    });

    container.appendChild(svg);
  }

  /* ---- banded curve ----------------------------------------------------
   * A line per trained series with its seed band, flat rules for the arms
   * that do not train, and a marker at each crossover budget. */
  function curveBand(container, figure) {
    container.textContent = "";
    var series = (figure.series || []).filter(function (s) {
      return s.points && s.points.length;
    });
    if (!series.length) {
      container.appendChild(el("div", "empty", figure.reason || "No curve on this split yet."));
      return;
    }

    var width = 760;
    /* Crossover captions get their own band above the plot rather than being
     * laid over it. Two low budgets sit close together on a log axis, so their
     * captions would otherwise overlap each other and the data. The band grows
     * with the number of captions instead of the plot shrinking to fit them. */
    var markerLine = 13;
    var captionBand = (figure.markers || []).reduce(function (total, m) {
      return total + (1 + ((m.lines || []).length)) * markerLine + 8;
    }, 0);
    var height = 360 + captionBand;
    var pad = { top: 22 + captionBand, right: 22, bottom: 56, left: 60 };
    var xs = [];
    var ys = [];
    series.forEach(function (s) {
      s.points.forEach(function (p) {
        xs.push(p.x);
        [p.y, p.lower, p.upper].forEach(function (v) {
          if (isNum(v)) ys.push(v);
        });
      });
    });
    (figure.references || []).forEach(function (r) {
      if (isNum(r.value)) ys.push(r.value);
    });

    var xMin = Math.max(1, Math.min.apply(null, xs));
    var xMax = Math.max.apply(null, xs);
    if (xMax <= xMin) xMax = xMin * 2;
    var yLo = Math.min.apply(null, ys);
    var yHi = Math.max.apply(null, ys);
    var span = Math.max(0.05, yHi - yLo);
    yLo = Math.max(0, yLo - span * 0.14);
    yHi = Math.min(1, yHi + span * 0.14);

    var logScale = figure.x_scale === "log";
    var lx0 = Math.log(xMin);
    var lx1 = Math.log(xMax);

    function px(x) {
      var t = logScale
        ? (Math.log(Math.max(1, x)) - lx0) / (lx1 - lx0)
        : (x - xMin) / (xMax - xMin);
      return pad.left + t * (width - pad.left - pad.right);
    }
    function py(y) {
      return height - pad.bottom - ((y - yLo) / (yHi - yLo)) * (height - pad.top - pad.bottom);
    }

    var svg = svgRoot(width, height, figure.title);

    for (var i = 0; i <= 4; i++) {
      var v = yLo + ((yHi - yLo) * i) / 4;
      svg.appendChild(
        node("line", {
          x1: pad.left, x2: width - pad.right, y1: py(v), y2: py(v),
          stroke: "var(--grid)", "stroke-width": 1
        })
      );
      svg.appendChild(axisText(pad.left - 10, py(v) + 4, fmt(v, 2), "end"));
    }
    var yTitle = node("text", {
      x: -(pad.top + (height - pad.bottom)) / 2, y: 14,
      transform: "rotate(-90)", "text-anchor": "middle",
      "font-size": 11, fill: "var(--muted)"
    });
    yTitle.textContent = figure.value_label;
    svg.appendChild(yTitle);

    /* Crossover rules, grouped by budget so several crossings at one budget
     * share a single caption instead of stacking on top of each other -- and
     * each budget's caption block is stepped down the plot, because two low
     * budgets sit close together on a log axis and their captions would
     * otherwise overlap and become unreadable. */
    var markerTop = 14;
    (figure.markers || []).forEach(function (m) {
      var captions = [m.label].concat(m.lines || []);
      svg.appendChild(
        node("line", {
          x1: px(m.x), x2: px(m.x), y1: markerTop - 10, y2: height - pad.bottom,
          stroke: "var(--line-strong)", "stroke-width": 1, "stroke-dasharray": "2 4"
        })
      );
      captions.forEach(function (line, idx) {
        var y = markerTop + idx * markerLine;
        svg.appendChild(axisText(px(m.x) + 6, y, line, "start", 10.5));
      });
      markerTop += captions.length * markerLine + 8;
    });

    (figure.references || []).forEach(function (r) {
      if (!isNum(r.value)) return;
      var colour = roleColour(r.role);
      svg.appendChild(
        node("line", {
          x1: pad.left, x2: width - pad.right, y1: py(r.value), y2: py(r.value),
          stroke: colour, "stroke-width": 2, "stroke-dasharray": "7 5"
        })
      );
      var label = node("text", {
        x: width - pad.right, y: py(r.value) - 7, "text-anchor": "end",
        "font-size": 11.5, fill: "var(--text)"
      });
      label.textContent = r.short + ", zero-shot · " + fmt(r.value, 3);
      svg.appendChild(label);
    });

    series.forEach(function (s) {
      var colour = roleColour(s.role);
      var banded = s.points.filter(function (p) {
        return isNum(p.lower) && isNum(p.upper);
      });
      if (banded.length > 1) {
        var top = banded.map(function (p) {
          return px(p.x).toFixed(1) + " " + py(p.upper).toFixed(1);
        });
        var bottom = banded
          .slice()
          .reverse()
          .map(function (p) {
            return px(p.x).toFixed(1) + " " + py(p.lower).toFixed(1);
          });
        svg.appendChild(
          node("path", {
            d: "M" + top.join(" L") + " L" + bottom.join(" L") + " Z",
            fill: colour, opacity: "var(--band-alpha)", stroke: "none"
          })
        );
      }
      svg.appendChild(
        node("path", {
          d: s.points
            .map(function (p, i) {
              return (i === 0 ? "M" : "L") + px(p.x).toFixed(1) + " " + py(p.y).toFixed(1);
            })
            .join(" "),
          fill: "none", stroke: colour, "stroke-width": 2,
          "stroke-dasharray": s.dash || "none",
          "stroke-linecap": "round", "stroke-linejoin": "round"
        })
      );
      s.points.forEach(function (p) {
        var marker = node("circle", { cx: px(p.x), cy: py(p.y), r: 4.5, fill: colour });
        var lines = [
          figure.x_label + " " + p.x,
          figure.value_label + " " + fmt(p.y, 3),
          "95% band " + fmt(p.lower, 3) + " to " + fmt(p.upper, 3)
        ];
        Object.keys(p.extra || {}).forEach(function (k) {
          lines.push(k + " " + p.extra[k]);
        });
        bindTip(marker, s.label, lines);
        svg.appendChild(marker);
      });
    });

    var seen = {};
    xs.forEach(function (x) {
      if (seen[x]) return;
      seen[x] = true;
      svg.appendChild(axisText(px(x), height - pad.bottom + 18, String(x)));
    });
    svg.appendChild(
      axisText((pad.left + width - pad.right) / 2, height - 10, figure.x_label)
    );

    container.appendChild(svg);
    container.appendChild(legendFor(series, figure.references));
  }

  /* ---- small multiples -------------------------------------------------
   * One panel per x value, every panel on the same domain so the comparison
   * is made by position. Bars, because each panel is a set of independent
   * proportions rather than a series over a continuum. */
  function smallMultiples(container, figure) {
    container.textContent = "";
    var series = (figure.series || []).filter(function (s) {
      return s.points && s.points.length;
    });
    var panels = figure.panels || [];
    if (!series.length || !panels.length) {
      container.appendChild(el("div", "empty", "No scored fault groups on this split yet."));
      return;
    }

    var grid = el("div", "multiples");
    panels.forEach(function (panel) {
      var cell = el("div", "multiple");
      cell.appendChild(el("h4", null, panel.label));
      cell.appendChild(el("p", "panel-n", panel.n + " false successes"));

      var rows = series
        .map(function (s) {
          var point = s.points.filter(function (p) {
            return p.panel === panel.key;
          })[0];
          return point ? { series: s, point: point } : null;
        })
        .filter(Boolean);

      var barH = 26;
      var padLeft = 92;
      var padRight = 12;
      var width = 320;
      var height = rows.length * barH + 26;
      var lo = figure.domain ? figure.domain[0] : 0;
      var hi = figure.domain ? figure.domain[1] : 1;

      function px(v) {
        return padLeft + ((v - lo) / (hi - lo)) * (width - padLeft - padRight);
      }

      var svg = svgRoot(width, height, panel.label);
      [0, 0.5, 1].forEach(function (frac) {
        var v = lo + (hi - lo) * frac;
        svg.appendChild(
          node("line", {
            x1: px(v), x2: px(v), y1: 0, y2: rows.length * barH,
            stroke: "var(--grid)", "stroke-width": 1
          })
        );
        svg.appendChild(
          axisText(px(v), rows.length * barH + 16, Math.round(frac * 100) + "%")
        );
      });

      rows.forEach(function (row, idx) {
        var cy = idx * barH + barH / 2;
        var colour = roleColour(row.series.role);
        var name = node("text", {
          x: 0, y: cy + 4, "font-size": 11.5, fill: "var(--text)"
        });
        name.textContent = row.series.short;
        svg.appendChild(name);

        if (!isNum(row.point.y)) {
          svg.appendChild(
            Object.assign(axisText(padLeft, cy + 4, "not scored", "start", 11), {})
          );
          return;
        }
        /* 4px rounded data-end on the baseline. */
        svg.appendChild(
          node("rect", {
            x: px(lo), y: cy - 5, width: Math.max(2, px(row.point.y) - px(lo)), height: 10,
            rx: 4, fill: colour
          })
        );
        if (isNum(row.point.lower) && isNum(row.point.upper)) {
          svg.appendChild(
            node("line", {
              x1: px(row.point.lower), x2: px(row.point.upper), y1: cy, y2: cy,
              stroke: "var(--text)", "stroke-width": 1.4, opacity: 0.55
            })
          );
        }
        var hit = node("rect", {
          x: 0, y: cy - barH / 2, width: width, height: barH, fill: "transparent"
        });
        bindTip(hit, panel.label + " · " + row.series.label, [
          "Detection rate " + Math.round(row.point.y * 100) + "%",
          "95% CI " + Math.round(row.point.lower * 100) + "% to " + Math.round(row.point.upper * 100) + "%",
          "Caught " + (row.point.extra || {}).Caught
        ]);
        svg.appendChild(hit);
      });
      cell.appendChild(svg);
      grid.appendChild(cell);
    });

    container.appendChild(grid);
    container.appendChild(legendFor(series, []));
  }

  /* ---- simple lines (measured series, no interval) ---------------------- */
  function lines(container, figure) {
    container.textContent = "";
    var series = (figure.series || []).filter(function (s) {
      return s.points && s.points.length;
    });
    if (!series.length) {
      container.appendChild(el("div", "empty", "No latency arms recorded."));
      return;
    }
    var singleArm = series[0].points.length < 2;
    if (singleArm) {
      /* One concurrency arm is not a line; a line through one point implies a
       * trend that was never measured. Show the arm's percentiles as bars. */
      var bars = {
        id: figure.id,
        kind: "interval-dots",
        title: figure.title,
        value_label: figure.value_label,
        digits: 0,
        domain: null,
        series: series.map(function (s) {
          return Object.assign({}, s, {
            group: "Concurrency " + s.points[0].x,
            points: [Object.assign({}, s.points[0], { lower: null, upper: null })]
          });
        })
      };
      intervalDots(container, bars);
      return;
    }

    var width = 640;
    var height = 260;
    var pad = { top: 18, right: 18, bottom: 48, left: 58 };
    var xs = [];
    var ys = [];
    series.forEach(function (s) {
      s.points.forEach(function (p) {
        xs.push(p.x);
        if (isNum(p.y)) ys.push(p.y);
      });
    });
    var xMin = Math.min.apply(null, xs);
    var xMax = Math.max.apply(null, xs);
    if (xMax === xMin) xMax = xMin + 1;
    var yMax = Math.max.apply(null, ys) * 1.12 || 1;

    function px(x) {
      return pad.left + ((x - xMin) / (xMax - xMin)) * (width - pad.left - pad.right);
    }
    function py(y) {
      return height - pad.bottom - (y / yMax) * (height - pad.top - pad.bottom);
    }

    var svg = svgRoot(width, height, figure.title);
    [0, 0.5, 1].forEach(function (frac) {
      svg.appendChild(
        node("line", {
          x1: pad.left, x2: width - pad.right, y1: py(yMax * frac), y2: py(yMax * frac),
          stroke: "var(--grid)", "stroke-width": 1
        })
      );
      svg.appendChild(
        axisText(pad.left - 10, py(yMax * frac) + 4, Math.round(yMax * frac), "end")
      );
    });
    series.forEach(function (s) {
      var colour = roleColour(s.role);
      svg.appendChild(
        node("path", {
          d: s.points
            .map(function (p, i) {
              return (i === 0 ? "M" : "L") + px(p.x).toFixed(1) + " " + py(p.y).toFixed(1);
            })
            .join(" "),
          fill: "none", stroke: colour, "stroke-width": 2,
          "stroke-linecap": "round", "stroke-linejoin": "round"
        })
      );
      s.points.forEach(function (p) {
        var marker = node("circle", { cx: px(p.x), cy: py(p.y), r: 4.5, fill: colour });
        bindTip(marker, s.label, [
          figure.x_label + " " + p.x,
          figure.value_label + " " + fmt(p.y, 0)
        ]);
        svg.appendChild(marker);
      });
    });
    var seen = {};
    xs.forEach(function (x) {
      if (seen[x]) return;
      seen[x] = true;
      svg.appendChild(axisText(px(x), height - pad.bottom + 18, String(x)));
    });
    svg.appendChild(axisText((pad.left + width - pad.right) / 2, height - 8, figure.x_label));
    container.appendChild(svg);
    container.appendChild(legendFor(series, []));
  }

  /* A legend for two or more series. Series of four or fewer are also direct-
   * labelled on the chart; the legend stays because the brief asks for one
   * whenever there is more than one series, and because it is the fastest way
   * to check which role a hue stands for. */
  function legendFor(series, references) {
    var wrap = el("div", "legend");
    var items = (series || []).concat(references || []);
    if (items.length < 2) return wrap;
    (series || []).forEach(function (s) {
      var item = el("span");
      var swatch = el("i");
      /* Series that share a hue are told apart by their dash pattern, so the
       * swatch has to carry it too or the legend loses the distinction. */
      swatch.style.background = s.dash
        ? "repeating-linear-gradient(90deg, " + roleColour(s.role) +
          " 0 5px, transparent 5px 8px)"
        : roleColour(s.role);
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(s.label));
      wrap.appendChild(item);
    });
    (references || []).forEach(function (r) {
      var item = el("span");
      var swatch = el("i", "dashed");
      swatch.style.color = roleColour(r.role);
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(r.label + " (zero-shot, flat)"));
      wrap.appendChild(item);
    });
    return wrap;
  }

  /* ---- table view ------------------------------------------------------
   * Rendered from figure.table, which the server derives from the same points
   * the chart draws. Never re-derived here: two derivations are two chances to
   * disagree. */
  function tableFor(figure) {
    var details = el("details", "fig-table");
    var summary = el("summary", null, "Table behind this figure");
    details.appendChild(summary);
    var wrap = el("div", "table-wrap");
    var table = document.createElement("table");
    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");
    var lead = figure.table.leading_columns === undefined ? 2 : figure.table.leading_columns;
    (figure.table.columns || []).forEach(function (column, i) {
      headRow.appendChild(el("th", i > lead ? "num" : null, column));
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    (figure.table.rows || []).forEach(function (row) {
      var tr = document.createElement("tr");
      row.forEach(function (cell, i) {
        tr.appendChild(el("td", i > lead ? "num" : null, cell));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    details.appendChild(wrap);
    if (figure.table.note) {
      details.appendChild(el("p", "table-note", figure.table.note));
    }
    return details;
  }

  var RENDERERS = {
    "interval-dots": intervalDots,
    "curve-band": curveBand,
    "small-multiples": smallMultiples,
    lines: lines
  };

  /* Render a whole figure: title, caption, chart, legend, table. */
  function render(host, figure) {
    host.textContent = "";
    if (!figure) {
      host.appendChild(el("div", "empty", "Figure unavailable."));
      return;
    }
    host.appendChild(el("p", "fig-title", figure.title));
    host.appendChild(el("p", "fig-caption", figure.caption));
    var canvas = el("div", "fig-canvas");
    (RENDERERS[figure.kind] || intervalDots)(canvas, figure);
    host.appendChild(canvas);
    host.appendChild(tableFor(figure));
  }

  global.Charts = { render: render, roleColour: roleColour, fmt: fmt };
})(window);
