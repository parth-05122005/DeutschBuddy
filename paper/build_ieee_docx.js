// Build the conference paper as an IEEE two-column Word file from paper.md.
//
// Run (from the project folder):
//   cd paper && npm install && node build_ieee_docx.js
// Output: "DeutschBuddy Conference Paper (IEEE).docx" in the project folder.
//
// Layout follows the IEEE A4 conference template: Times New Roman, 24 pt title, 10 pt body,
// two columns of 88.9 mm with 4.2 mm spacing, small-caps section headings, "TABLE I" captions
// above tables, "Fig. 1." captions below figures, 8 pt numbered references.
// Placeholders such as [XX] and [Department] are highlighted in yellow so they are easy to find.

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, HighlightColor, ImageRun, LevelFormat, Packer, Paragraph,
  SectionType, Table, TableCell, TableRow, TabStopType, TextRun, WidthType,
} = require("docx");

const HERE = __dirname;
const OUT = path.join(HERE, "..", "DeutschBuddy Conference Paper (IEEE).docx");
const MM = 56.7; // twips per millimetre

const PAGE = {
  size: { width: 11906, height: 16838 }, // A4
  margin: { top: Math.round(19 * MM), bottom: Math.round(43 * MM), left: Math.round(14.32 * MM), right: Math.round(14.32 * MM) },
};
const COLUMN_WIDTH = Math.round(88.9 * MM); // 5041 twips
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];

// ---------- Inline formatting: **bold**, *italic*, placeholders highlighted ----------

const PLACEHOLDER = /\[(?!\d)[^\]]+\]/; // [XX], [Department], [Describe ...] but not citations like [3]

function runs(text, base = {}) {
  const out = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|\[(?!\d)[^\]]+\])/g;
  let last = 0;
  for (const m of text.matchAll(pattern)) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    const token = m[0];
    if (token.startsWith("**")) out.push(...runs(token.slice(2, -2), { ...base, bold: true }));
    else if (token.startsWith("*")) out.push(...runs(token.slice(1, -1), { ...base, italics: true }));
    else out.push(new TextRun({ ...base, text: token, highlight: HighlightColor.YELLOW }));
    last = m.index + token.length;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out;
}

const body = (text, opts = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { firstLine: opts.noIndent ? 0 : 288 },
  spacing: { after: 0, line: 240 }, children: runs(text, opts.run || {}),
});

// ---------- Block builders ----------

function sectionHeading(text) {
  // "I. Introduction" → small caps, centred, like the IEEE template
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, smallCaps: true })], keepNext: true,
  });
}

function subHeading(text) {
  return new Paragraph({
    spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new TextRun({ text, italics: true })],
  });
}

function captionedTable(label, title, rows) {
  const header = rows[0];
  const columns = header.length;
  // First column wider, the rest share the remaining width
  const first = Math.round(COLUMN_WIDTH * (columns > 5 ? 0.27 : 0.34));
  const other = Math.floor((COLUMN_WIDTH - first) / (columns - 1));
  const widths = [first, ...Array(columns - 1).fill(other)];
  widths[0] += COLUMN_WIDTH - widths.reduce((a, b) => a + b, 0);

  const line = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const tableRows = rows.map((cells, r) => new TableRow({
    tableHeader: r === 0,
    children: cells.map((cell, c) => new TableCell({
      width: { size: widths[c], type: WidthType.DXA },
      margins: { top: 20, bottom: 20, left: 50, right: 50 },
      borders: { top: r === 0 ? line : none, bottom: r === 0 || r === rows.length - 1 ? line : none, left: none, right: none },
      children: [new Paragraph({
        alignment: c === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: runs(cell, { size: 14, bold: r === 0 }),
      })],
    })),
  }));

  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160 }, keepNext: true,
      children: [new TextRun({ text: label, size: 16, smallCaps: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
      children: [new TextRun({ text: title, size: 16, smallCaps: true })] }),
    new Table({ width: { size: COLUMN_WIDTH, type: WidthType.DXA }, columnWidths: widths, rows: tableRows }),
    new Paragraph({ spacing: { after: 80 }, children: [] }),
  ];
}

function figure(caption, file) {
  const data = fs.readFileSync(path.join(HERE, file));
  // PNG size from the header, scaled to the column width (3.4 in = 326 px at 96 dpi)
  const w = data.readUInt32BE(16), h = data.readUInt32BE(20);
  const width = 326, height = Math.round(width * h / w);
  const [label, ...rest] = caption.split(". ");
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width, height } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 120 },
      children: [new TextRun({ text: `${label}. `, size: 16 }), ...runs(rest.join(". "), { size: 16 })] }),
  ];
}

function reference(text, n) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, indent: { left: 360, hanging: 360 },
    tabStops: [{ type: TabStopType.LEFT, position: 360 }], spacing: { after: 20 },
    children: [new TextRun({ text: `[${n}]\t`, size: 16 }), ...runs(text, { size: 16 })],
  });
}

// ---------- Parse paper.md ----------

function parse(markdown) {
  const lines = markdown.replace(/\r/g, "").split("\n");
  const front = [], main = [];
  const authors = [];
  let title = "", mode = "body", para = [], i = 0;

  const flush = () => {
    if (!para.length) return;
    const text = para.join(" ");
    if (mode === "abstract") main.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
      children: [new TextRun({ text: "Abstract", bold: true, italics: true, size: 18 }),
                 new TextRun({ text: "—", bold: true, size: 18 }), ...runs(text, { bold: true, size: 18 })] }));
    else if (mode === "index") main.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
      children: [new TextRun({ text: "Index Terms", bold: true, italics: true, size: 18 }),
                 new TextRun({ text: "—", bold: true, size: 18 }), ...runs(text, { bold: true, size: 18 })] }));
    else main.push(body(text));
    para = [];
  };

  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("# ")) { title = line.slice(2).trim(); i++; continue; }
    if (line.startsWith("@author ")) { authors.push(line.slice(8).split("|").map((s) => s.trim())); i++; continue; }
    if (line.startsWith("## ")) {
      flush();
      const name = line.slice(3).trim();
      if (name === "Abstract") mode = "abstract";
      else if (name === "Index Terms") mode = "index";
      else { mode = name === "References" ? "refs" : "body"; main.push(sectionHeading(name)); }
      i++; continue;
    }
    if (line.startsWith("### ")) { flush(); main.push(subHeading(line.slice(4).trim())); i++; continue; }
    if (line.startsWith("Table: ")) {
      flush();
      const [label, caption] = line.slice(7).split("|").map((s) => s.trim());
      const rows = [];
      i++;
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = lines[i].split("|").slice(1, -1).map((s) => s.trim());
        if (!cells.every((c) => /^-+$/.test(c))) rows.push(cells);
        i++;
      }
      main.push(...captionedTable(label, caption, rows));
      continue;
    }
    const img = line.match(/^!\[(.+)\]\((.+)\)$/);
    if (img) { flush(); main.push(...figure(img[1], img[2])); i++; continue; }
    const numbered = line.match(/^(\d+)\. (.*)$/);
    if (numbered && mode === "refs") { main.push(reference(numbered[2], numbered[1])); i++; continue; }
    if (numbered) {
      flush();
      main.push(new Paragraph({ numbering: { reference: "numbered", level: 0 }, alignment: AlignmentType.JUSTIFIED,
        children: runs(numbered[2]) }));
      i++; continue;
    }
    if (line.startsWith("- ")) {
      flush();
      main.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
        children: runs(line.slice(2)) }));
      i++; continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    para.push(line.trim());
    i++;
  }
  flush();

  front.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: title, size: 48 })] }));
  const authorCell = (a) => new TableCell({
    width: { size: Math.floor((PAGE.size.width - PAGE.margin.left - PAGE.margin.right) / authors.length), type: WidthType.DXA },
    borders: { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
               left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
    children: a.map((part, k) => new Paragraph({ alignment: AlignmentType.CENTER,
      children: runs(part, k === 0 ? { size: 22 } : { size: 20, italics: !part.includes("@") }) })),
  });
  const fullWidth = PAGE.size.width - PAGE.margin.left - PAGE.margin.right;
  front.push(new Table({ width: { size: fullWidth, type: WidthType.DXA },
    columnWidths: authors.map(() => Math.floor(fullWidth / authors.length)),
    rows: [new TableRow({ children: authors.map(authorCell) })] }));
  front.push(new Paragraph({ spacing: { after: 240 }, children: [] }));
  return { front, main };
}

// ---------- Assemble ----------

const { front, main } = parse(fs.readFileSync(path.join(HERE, "paper.md"), "utf8"));

const doc = new Document({
  creator: "Parth Khandelwal",
  title: "Does Retrieval Make LLM Grammar Explanations Trustworthy?",
  styles: { default: { document: { run: { font: "Times New Roman", size: 20 } } } },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 288, hanging: 180 } } } }] },
    { reference: "numbered", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 288, hanging: 216 } } } }] },
  ] },
  sections: [
    { properties: { page: PAGE }, children: front },
    { properties: { type: SectionType.CONTINUOUS, page: PAGE,
        column: { count: 2, space: Math.round(4.22 * MM), equalWidth: true } }, children: main },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT, buffer);
  console.log(`Wrote ${OUT}`);
});
