// Generate manuscript.docx from docs/manuscript_federated_ehr.md
// Target format matches reviewer-style Word docs: Calibri 12pt, 0.7" margins,
// 1.38 line spacing, justified body, bold figure/table callouts, figures inline.
//
// Run: NODE_PATH=$(npm root -g) node manuscript/build/generate_manuscript.js

const {
  Document, Packer, Paragraph, TextRun, AlignmentType,
  ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle, HeightRule,
} = require('docx');
const fs = require('fs');
const path = require('path');

const REPO = path.resolve(__dirname, '..', '..');
const MD_PATH = path.join(REPO, 'docs/manuscript_federated_ehr.md');
const FIG_DIR = path.join(REPO, 'figures');
const OUT_DIR = path.join(REPO, 'manuscript', 'build');
const OUT_PATH = path.join(OUT_DIR, 'manuscript_draft_v1.docx');

const BODY_FONT = 'Calibri';
const BODY_SIZE = 24;   // 12pt (half-points)
const TITLE_SIZE = 40;  // 20pt
const H1_SIZE = 36;     // 18pt
const H2_SIZE = 28;     // 14pt
const H3_SIZE = 26;     // 13pt
const CAPTION_SIZE = 22; // 11pt

// Figures to embed (first reference in text)
// The md source uses "Figure 1", "Figure 2", "Figure 3" callouts.
// We insert each figure image AFTER its figure-legend paragraph (the bold
// "Figure N." paragraph in the Figure legends section).
const FIGURES = {
  1: { file: 'paper_fig1_strategies.png',           widthInches: 6.7 },
  2: { file: 'paper_fig2_main_results.png',         widthInches: 6.7 },
  3: { file: 'paper_fig3_alpha_robustness.png',     widthInches: 6.7 },
  4: { file: 'paper_fig4_calibration.png',          widthInches: 6.7 },
};

// ---------- helpers ----------

const EMU_PER_INCH = 914400;

function boldifyCallouts(text) {
  const segments = [];
  const re = /(Figure|Table)\s+S?\d+[a-z]*(?:[,\-][a-z]+)*(?:-?[a-z]+)?/g;
  let lastEnd = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > lastEnd) segments.push({ text: text.slice(lastEnd, m.index), bold: false });
    segments.push({ text: m[0], bold: true });
    lastEnd = m.index + m[0].length;
  }
  if (lastEnd < text.length) segments.push({ text: text.slice(lastEnd), bold: false });
  return segments;
}

function parseInline(text, { inheritBold = false, inheritItalic = false } = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^\^]+\^|[^*^]+|\*|\^)/g;
  let m;
  const pushRun = (t, opts = {}) => {
    t = t.replace(/\\([*^])/g, '$1');
    const bold = opts.bold || inheritBold;
    const italics = opts.italics || inheritItalic;
    if (!bold && !italics && !opts.superScript) {
      // Auto-bold figure/table callouts
      for (const s of boldifyCallouts(t)) {
        runs.push(new TextRun({
          text: s.text, bold: s.bold, font: BODY_FONT, size: BODY_SIZE, color: '000000',
        }));
      }
    } else {
      runs.push(new TextRun({
        text: t, bold, italics, superScript: opts.superScript,
        font: BODY_FONT, size: BODY_SIZE, color: '000000',
      }));
    }
  };
  while ((m = re.exec(text)) !== null) {
    const t = m[0];
    if (t.startsWith('**') && t.endsWith('**') && t.length > 4) pushRun(t.slice(2, -2), { bold: true });
    else if (t.startsWith('*') && t.endsWith('*') && t.length > 2) pushRun(t.slice(1, -1), { italics: true });
    else if (t.startsWith('^') && t.endsWith('^') && t.length > 2) pushRun(t.slice(1, -1), { superScript: true });
    else pushRun(t);
  }
  return runs;
}

function bodyParagraph(text) {
  return new Paragraph({
    spacing: { after: 120, line: 276, lineRule: 'auto' },
    alignment: AlignmentType.JUSTIFIED,
    children: parseInline(text),
  });
}

function makeImageParagraph(filepath, widthInches) {
  const buffer = fs.readFileSync(filepath);
  const ext = path.extname(filepath).toLowerCase();
  const type = ext === '.png' ? 'png' : (ext === '.jpg' || ext === '.jpeg') ? 'jpg' : 'png';
  // Detect aspect ratio from the image (simple: use fixed aspect based on filename)
  const ratios = {
    'paper_fig1_strategies.png': 16 / 9,
    'paper_fig2_main_results.png': 22 / 20,
    'paper_fig3_alpha_robustness.png': 16 / 6.5,
    'paper_fig4_calibration.png': 22 / 5.2,
  };
  const ar = ratios[path.basename(filepath)] || 4 / 3;
  const heightInches = widthInches / ar;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 180, after: 120 },
    children: [
      new ImageRun({
        data: buffer,
        type: type,
        transformation: { width: widthInches * 96, height: heightInches * 96 },
      }),
    ],
  });
}

// Parse a markdown table block into a docx Table
function parseMarkdownTable(lines, startIdx) {
  // A markdown table starts with '|' and has a separator line like |---|
  const rows = [];
  let i = startIdx;
  while (i < lines.length && lines[i].trim().startsWith('|')) {
    rows.push(lines[i].trim());
    i++;
  }
  if (rows.length < 2) return { table: null, consumed: 0 };
  // Remove separator line (rows[1])
  const headerRaw = rows[0];
  const bodyRaw = rows.slice(2);
  const splitCells = (row) => row.slice(1, -1).split('|').map(c => c.trim());
  const headerCells = splitCells(headerRaw);
  const bodyRows = bodyRaw.map(splitCells);

  const tableWidthTwips = 9360; // 6.5 inch usable width at 0.7" margins (wider than 9000 default)

  const mkCell = (text, isHeader) => {
    const runs = parseInline(text, { inheritBold: isHeader });
    return new TableCell({
      children: [new Paragraph({
        children: runs,
        spacing: { after: 40 },
      })],
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
    });
  };

  const trs = [
    new TableRow({
      tableHeader: true,
      children: headerCells.map(c => mkCell(c, true)),
    }),
    ...bodyRows.map(cells => new TableRow({
      children: cells.map(c => mkCell(c, false)),
    })),
  ];

  const table = new Table({
    rows: trs,
    width: { size: tableWidthTwips, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: '808080' },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: '808080' },
      left: { style: BorderStyle.SINGLE, size: 4, color: '808080' },
      right: { style: BorderStyle.SINGLE, size: 4, color: '808080' },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' },
    },
  });

  return { table, consumed: rows.length };
}

// ---------- main ----------

const md = fs.readFileSync(MD_PATH, 'utf8');
const lines = md.split('\n');
const children = [];

let i = 0;
let insertedFigures = new Set(); // figure numbers already inserted

while (i < lines.length) {
  const line = lines[i];

  // Skip blank lines and horizontal rules
  if (line.trim() === '' || line.trim() === '---') {
    i++;
    continue;
  }

  // Title (# )
  if (line.startsWith('# ')) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 240 },
      children: [new TextRun({
        text: line.slice(2).trim(), bold: true,
        font: BODY_FONT, size: TITLE_SIZE, color: '000000',
      })],
    }));
    i++;
    continue;
  }

  // H1 (## )
  if (line.startsWith('## ')) {
    children.push(new Paragraph({
      spacing: { before: 280, after: 140 },
      children: [new TextRun({
        text: line.slice(3).trim().toUpperCase(), bold: true,
        font: BODY_FONT, size: H1_SIZE, color: '000000',
      })],
    }));
    i++;
    continue;
  }

  // H2 (### )
  if (line.startsWith('### ')) {
    children.push(new Paragraph({
      spacing: { before: 200, after: 100 },
      children: [new TextRun({
        text: line.slice(4).trim(), bold: true,
        font: BODY_FONT, size: H2_SIZE, color: '000000',
      })],
    }));
    i++;
    continue;
  }

  // Markdown table block
  if (line.trim().startsWith('|')) {
    const { table, consumed } = parseMarkdownTable(lines, i);
    if (table) {
      children.push(table);
      // Blank paragraph after table for spacing
      children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      i += consumed;
      continue;
    }
  }

  // Regular paragraph (may span multiple lines — we treat each line as its own paragraph
  // since our source uses blank-line-separated paragraphs).
  children.push(bodyParagraph(line));

  // After inserting a Figure legend paragraph like "**Figure N. ...**", embed the figure image
  const figMatch = line.match(/^\*\*Figure (\d+)\. /);
  if (figMatch) {
    const figNum = parseInt(figMatch[1], 10);
    if (FIGURES[figNum] && !insertedFigures.has(figNum)) {
      const figPath = path.join(FIG_DIR, FIGURES[figNum].file);
      if (fs.existsSync(figPath)) {
        // Put image BEFORE the caption: we just pushed the caption; insert image
        // at position len-1 so image is above caption.
        const captionPara = children.pop();
        children.push(makeImageParagraph(figPath, FIGURES[figNum].widthInches));
        children.push(captionPara);
        insertedFigures.add(figNum);
      }
    }
  }

  i++;
}

// Build doc
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: BODY_FONT, size: BODY_SIZE, color: '000000' } },
    },
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 1008, right: 1008, bottom: 1008, left: 1008 }, // 0.7"
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_PATH, buffer);
  console.log('Saved:', OUT_PATH);
  console.log('Size:', (buffer.length / 1024).toFixed(1), 'KB');
  console.log('Figures embedded:', [...insertedFigures].sort());

  // Auto-reload: if the docx is currently open in Word as the active document,
  // close it (without saving, since we just overwrote the file) and reopen
  // the fresh copy. Modern Word on macOS restricts AppleScript to the active
  // document only, which is the common case anyway.
  const { spawnSync } = require('child_process');
  const fileName = path.basename(OUT_PATH);
  const osascript = `try
  tell application "Microsoft Word"
    set activeName to name of active document
    if activeName is "${fileName}" then
      close active document saving no
      open POSIX file "${OUT_PATH}"
      return "reloaded"
    end if
  end tell
on error
  return "not-open"
end try
return "not-open"`;
  try {
    const res = spawnSync('osascript', ['-e', osascript], {
      timeout: 5000, encoding: 'utf8',
    });
    const out = (res.stdout || '').trim();
    if (out === 'reloaded') {
      console.log('(Word document reloaded)');
    } else if (out && out !== 'not-open') {
      console.log('(auto-reload result: ' + out + ')');
    }
  } catch (e) {
    // Silently ignore if Word isn't running
  }
});
