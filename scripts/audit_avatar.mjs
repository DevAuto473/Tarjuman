#!/usr/bin/env node
/**
 * قياسُ ملفّ GLB: أين تذهب البايتات فعلاً.
 *
 * كُتب هذا قبل `compress_avatar.mjs` لأنّ علاجَ الحجم يختلف باختلاف سببه:
 * إن كانت الصورُ هي الثقل فالتصغيرُ يربح، وإن كانت الرؤوسُ فضغطُ الهندسة.
 * في `last11.glb` كان الجواب 93.5% صوراً — فبُني السكربتُ الآخر على ذلك.
 *
 * الاستعمال:  npm run audit:avatar -- [tarjuman/public/last11.glb]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FILE = path.resolve(ROOT, process.argv[2] || 'tarjuman/public/last11.glb');

const buf = fs.readFileSync(FILE);
if (buf.readUInt32LE(0) !== 0x46546c67) { console.error('ليس ملفّ GLB'); process.exit(1); }

let off = 12, json = null, binLen = 0, binOff = 0;
while (off < buf.length) {
  const len = buf.readUInt32LE(off), type = buf.readUInt32LE(off + 4);
  off += 8;
  if (type === 0x4e4f534a) json = JSON.parse(buf.subarray(off, off + len).toString('utf8'));
  else if (type === 0x004e4942) { binLen = len; binOff = off; }
  off += len;
}

const { bufferViews = [], accessors = [], meshes = [], images = [], nodes = [], skins = [] } = json;

// تصنيفُ كلّ accessor بدوره، ثمّ نسبةُ كلّ دورٍ من البايتات.
const cat = new Array(accessors.length).fill('other');
for (const m of meshes) for (const p of m.primitives || []) {
  for (const [sem, i] of Object.entries(p.attributes || {})) cat[i] = 'attr:' + sem.split('_')[0];
  if (p.indices != null) cat[p.indices] = 'indices';
  for (const t of p.targets || []) for (const [sem, i] of Object.entries(t)) cat[i] = 'morph:' + sem;
}
for (const a of json.animations || []) for (const s of a.samplers || [])
  for (const k of ['input', 'output']) if (s[k] != null) cat[s[k]] = 'animation';
for (const s of skins) if (s.inverseBindMatrices != null) cat[s.inverseBindMatrices] = 'inverseBindMatrices';

const owner = new Map();
for (const im of images) if (im.bufferView != null && !owner.has(im.bufferView))
  owner.set(im.bufferView, 'image:' + (im.mimeType || '?'));
accessors.forEach((a, i) => { if (a.bufferView != null && !owner.has(a.bufferView)) owner.set(a.bufferView, cat[i]); });

const tally = new Map();
bufferViews.forEach((bv, i) => {
  const k = owner.get(i) ?? 'UNCLAIMED';
  tally.set(k, (tally.get(k) || 0) + bv.byteLength);
});

const mb = (n) => (n / 1048576).toFixed(2);
console.log(`${path.relative(ROOT, FILE)}  —  ${mb(buf.length)} MB  (BIN ${mb(binLen)} MB)\n`);
console.log('الفئة                          بايت            MB     %');
for (const [k, v] of [...tally].sort((a, b) => b[1] - a[1]))
  console.log(`${k.padEnd(28)} ${String(v).padStart(12)} ${mb(v).padStart(9)} ${(v / buf.length * 100).toFixed(1).padStart(6)}%`);

console.log(`\nصور: ${images.length}`);
for (const [i, im] of images.entries()) {
  const bv = bufferViews[im.bufferView];
  const s = binOff + (bv.byteOffset || 0);
  const w = buf.readUInt32BE(s + 16), h = buf.readUInt32BE(s + 20);   // ترويسةُ PNG
  const dims = im.mimeType === 'image/png' ? `${w}x${h}` : '(غير PNG)';
  console.log(`  [${i}] ${(im.name || '').padEnd(24)} ${im.mimeType}  ${dims}  ${mb(bv.byteLength)} MB`);
}

let tri = 0, vtx = 0;
for (const m of meshes) for (const p of m.primitives || []) {
  if (p.indices != null) tri += accessors[p.indices].count / 3;
  if (p.attributes?.POSITION != null) vtx += accessors[p.attributes.POSITION].count;
}
console.log(`\nشبكات: ${meshes.length}   عقد: ${nodes.length}   مفاصل: ${skins.map(s => s.joints.length).join(',')}`);
console.log(`مثلّثات: ${tri.toLocaleString()}   رؤوس: ${vtx.toLocaleString()}`);
