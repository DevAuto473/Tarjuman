#!/usr/bin/env node
/**
 * ضغطُ الأفاتار للرازبيري باي — سكربتٌ قابلٌ للإعادة.
 *
 * لِمَ هذا الملفّ موجود: `last11.glb` أربعون ميغابايت، و93.5% منها أربعُ
 * صور PNG بمقاس 4096×4096 تخصّ خامةً واحدةً (Ch36_Body) على شبكةٍ واحدةٍ
 * هي جسد Mixamo. الهندسةُ كلُّها 2.4 MB فقط. لذا الكسبُ في النسيج لا في
 * ضغط الرؤوس — قِيسَ ذلك بـ scripts/audit_avatar.mjs قبل كتابة هذا.
 *
 * ما يفعله:
 *   1. يصغّر كلّ نسيجٍ إلى --size (افتراضاً 1024) ويحوّله WebP.
 *      خريطةُ النتوء (normalTexture) تُضغط بجودةٍ أعلى لأنّ WebP الخاسر
 *      يترك تموّجاً مرئيّاً على الأسطح الملساء.
 *   2. اختيارياً --meshopt: ضغطُ الهندسة (EXT_meshopt_compression).
 *      مطفأٌ افتراضاً: الكسبُ ~1 MB مقابل خطرِ تكميمِ أوزان الهيكل.
 *   3. يتحقّق أنّ أسماء العقد الـ94 كلّها نجت — برمجيّاً بمقارنة القائمتين،
 *      لا بالعين. RobotStage.jsx و useSignPlayer.js ينادِيان العظام بالاسم،
 *      وعظمٌ مُعادُ التسمية يكسر تشغيلَ الإشارات بصمت.
 *
 * الاستعمال:  npm run compress:avatar -- [--size 1024] [--meshopt] [--quality 82]
 */
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { textureCompress, meshopt, prune, dedup } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import sharp from 'sharp';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC = path.join(ROOT, 'tarjuman', 'public');

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};
const has = (name) => argv.includes(`--${name}`);

const INPUT   = path.resolve(PUBLIC, flag('in', 'last11.glb'));
const OUTPUT  = path.resolve(PUBLIC, flag('out', 'last11.pi.glb'));
const SIZE    = Number(flag('size', 1024));
const QUALITY = Number(flag('quality', 82));
const MESHOPT = has('meshopt');

const mb = (n) => (n / 1048576).toFixed(2) + ' MB';

async function main() {
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;

  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      'meshopt.encoder': MeshoptEncoder,
      'meshopt.decoder': MeshoptDecoder,
    });

  const beforeBytes = (await fs.stat(INPUT)).size;
  console.log(`المصدر : ${path.basename(INPUT)}  ${mb(beforeBytes)}`);

  const doc = await io.read(INPUT);
  const namesBefore = doc.getRoot().listNodes().map((n) => n.getName());

  console.log('\nالنسيج قبل:');
  for (const t of doc.getRoot().listTextures()) {
    console.log(`  ${t.getName()}  ${t.getMimeType()}  ${mb(t.getImage().byteLength)}`);
  }

  // خريطةُ النتوء أوّلاً وبجودةٍ أعلى — ثمّ البقيّة.
  await doc.transform(
    textureCompress({
      encoder: sharp,
      targetFormat: 'webp',
      slots: /normalTexture/,
      resize: [SIZE, SIZE],
      quality: Math.min(100, QUALITY + 13),
      effort: 90,
    }),
    textureCompress({
      encoder: sharp,
      targetFormat: 'webp',
      resize: [SIZE, SIZE],
      quality: QUALITY,
      effort: 90,
    }),
    dedup(),
    prune({ keepAttributes: true, keepLeaves: true }),
    ...(MESHOPT ? [meshopt({ encoder: MeshoptEncoder, level: 'medium' })] : []),
  );

  console.log('\nالنسيج بعد:');
  for (const t of doc.getRoot().listTextures()) {
    console.log(`  ${t.getName()}  ${t.getMimeType()}  ${mb(t.getImage().byteLength)}`);
  }

  await io.write(OUTPUT, doc);
  const afterBytes = (await fs.stat(OUTPUT)).size;

  // ——— التحقّق: أسماء العقد ———
  const verifyDoc = await io.read(OUTPUT);
  const namesAfter = verifyDoc.getRoot().listNodes().map((n) => n.getName());
  const setAfter = new Set(namesAfter);
  const missing = namesBefore.filter((n) => !setAfter.has(n));

  const FACE_BONES = ['jaw', 'lid_up.L', 'lid_up.R', 'lid_lo.L', 'lid_lo.R', 'brow.L', 'brow.R'];
  const faceMissing = FACE_BONES.filter((b) => !setAfter.has(b));

  const skinsBefore = doc.getRoot().listSkins().map((s) => s.listJoints().length);
  const skinsAfter  = verifyDoc.getRoot().listSkins().map((s) => s.listJoints().length);

  console.log('\n—— التحقّق ——');
  console.log(`عقدٌ قبل/بعد      : ${namesBefore.length} / ${namesAfter.length}`);
  console.log(`أسماءٌ مفقودة      : ${missing.length ? missing.join(', ') : 'لا شيء ✅'}`);
  console.log(`عظامُ الوجه السبع  : ${faceMissing.length ? '✗ ' + faceMissing.join(', ') : 'كلُّها موجودة ✅'}`);
  console.log(`مفاصلُ الهيكل      : ${skinsBefore.join(',')} → ${skinsAfter.join(',')}`);
  console.log(`شبكاتٌ قبل/بعد     : ${doc.getRoot().listMeshes().length} / ${verifyDoc.getRoot().listMeshes().length}`);

  console.log('\n—— الحجم ——');
  console.log(`${mb(beforeBytes)} → ${mb(afterBytes)}  (${(100 - (afterBytes / beforeBytes) * 100).toFixed(1)}% أقلّ)`);
  console.log(`الهدف < 8 MB : ${afterBytes < 8 * 1048576 ? 'محقَّق ✅' : 'غيرُ محقَّق ✗'}`);
  console.log(`الناتج: ${OUTPUT}`);

  const ok = missing.length === 0 && faceMissing.length === 0 && afterBytes < 8 * 1048576;
  process.exit(ok ? 0 : 1);
}

main().catch((err) => { console.error(err); process.exit(1); });
