/**
 * CalibrationPanel.jsx — determine the rig's direction convention empirically
 * ===========================================================================
 * Two attempts at deriving the robot's poses from recorded data produced arms
 * in the wrong place. Both failed for the same reason: the maths assumed a
 * coordinate convention that could not be verified without seeing the result.
 *
 * This panel stops the guessing. It drives a single bone toward a KNOWN
 * direction and lets you say what actually happened on screen. One controlled
 * test replaces a chain of assumptions.
 *
 * How to use it
 * -------------
 *   1. Press "Arms UP". The arms should rise straight above the head.
 *   2. If they went somewhere else, press the button describing what you saw.
 *   3. The panel prints the axis mapping to apply in export_signs_3d.py.
 */

import React, { useState } from 'react';
import * as THREE from 'three';
import { X, Compass } from 'lucide-react';

// Directions in the space the exporter currently assumes:
// +x = screen right, +y = up, +z = toward the viewer.
const PROBES = [
  { id: 'up',      label: 'الذراعان لأعلى',   dir: [0, 1, 0],  expect: 'ترتفعان فوق الرأس' },
  { id: 'down',    label: 'الذراعان لأسفل',   dir: [0, -1, 0], expect: 'تتدلّيان بجانب الجسم' },
  { id: 'out',     label: 'الذراعان للخارج',  dir: [1, 0, 0],  expect: 'تمتدّان أفقياً يميناً' },
  { id: 'forward', label: 'الذراعان للأمام',  dir: [0, 0, 1],  expect: 'تشيران نحوك' },
];

// What the user might see instead — each maps to an axis correction.
const OUTCOMES = [
  { id: 'correct', label: 'صحيح — كما هو متوقع',  fix: null },
  { id: 'mirrored', label: 'معكوس يمين/يسار',      fix: 'MIRROR_X = True' },
  { id: 'flipped',  label: 'مقلوب رأساً على عقب',  fix: 'FLIP_Y = True' },
  { id: 'depth',    label: 'للأمام/للخلف بدل ذلك', fix: 'SWAP_YZ = True' },
];

export default function CalibrationPanel({ signerRef, onClose }) {
  const [active, setActive] = useState(null);
  const [verdicts, setVerdicts] = useState({});

  /** Pose both arms toward one direction and hold it. */
  const probe = (p) => {
    const player = signerRef.current;
    if (!player) return;
    setActive(p.id);

    const pose = {};
    for (const side of ['L', 'R']) {
      pose[`UpperArm.${side}`] = p.dir;
      pose[`LowerArm.${side}`] = p.dir;
    }
    // A long, two-key "sign" so the pose is reached and simply held while
    // being inspected.
    player.playSigns([{
      word: `calib:${p.id}`,
      sign: {
        duration: 4.0,
        format: 'directions',
        keys: [{ t: 0, pose }, { t: 1, pose }],
      },
    }]);
  };

  const record = (outcome) => {
    if (!active) return;
    setVerdicts((v) => ({ ...v, [active]: outcome }));
  };

  const fixes = [...new Set(
    Object.values(verdicts).map((o) => o.fix).filter(Boolean)
  )];

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4" dir="rtl">
      <div className="bg-[#111827] border border-white/10 rounded-2xl w-full max-w-lg p-5 flex flex-col gap-4 shadow-2xl">
        <div className="flex justify-between items-center border-b border-white/10 pb-3">
          <div className="flex items-center gap-2">
            <Compass className="w-5 h-5 text-amber-400" />
            <h3 className="text-xl font-bold text-white">معايرة اتجاهات الروبوت</h3>
          </div>
          <button onClick={onClose}
            className="text-white/60 hover:text-white p-1 rounded-lg hover:bg-white/10 cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        <p className="text-white/60 text-sm leading-relaxed">
          اضغط اتجاهاً وشاهد الروبوت، ثم أخبرني بما رأيته فعلاً.
          هذا يحدّد اصطلاح المحاور بتجربة واحدة بدل التخمين.
        </p>

        {/* Step 1 — drive a known direction */}
        <div>
          <div className="text-white/40 text-xs mb-2">١. اختر اتجاهاً</div>
          <div className="grid grid-cols-2 gap-2">
            {PROBES.map((p) => (
              <button key={p.id} onClick={() => probe(p)}
                className={`px-3 py-2.5 rounded-xl border text-right transition cursor-pointer ${active === p.id
                  ? 'bg-amber-500/20 border-amber-500 text-amber-100'
                  : 'bg-white/5 border-white/10 text-white/85 hover:bg-white/10'
                  }`}>
                <div className="font-bold text-sm">{p.label}</div>
                <div className="text-[11px] text-white/40">{p.expect}</div>
                {verdicts[p.id] && (
                  <div className="text-[11px] text-emerald-400 mt-1">
                    ✓ {verdicts[p.id].label}
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Step 2 — report what happened */}
        {active && (
          <div>
            <div className="text-white/40 text-xs mb-2">٢. ماذا حدث فعلاً؟</div>
            <div className="grid grid-cols-2 gap-2">
              {OUTCOMES.map((o) => (
                <button key={o.id} onClick={() => record(o)}
                  className="px-3 py-2 rounded-xl bg-white/5 border border-white/10
                             text-white/85 text-sm hover:bg-white/10 transition cursor-pointer">
                  {o.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Result */}
        {Object.keys(verdicts).length > 0 && (
          <div className="bg-black/40 border border-white/10 rounded-xl p-3">
            <div className="text-white/40 text-xs mb-2">النتيجة</div>
            {fixes.length === 0 ? (
              <p className="text-emerald-400 text-sm">
                الاصطلاح صحيح — لا حاجة لتعديل. لو استمرت الإشارات بالظهور خطأً،
                فالمشكلة في بيانات التسجيل لا في المحاور.
              </p>
            ) : (
              <>
                <p className="text-amber-300 text-sm mb-2">
                  عدّل هذه القيم في <code>export_signs_3d.py</code> ثم
                  أعد <code>npm run export3d</code>:
                </p>
                <pre className="text-emerald-300 text-xs bg-black/50 p-2 rounded overflow-x-auto">
{fixes.join('\n')}
                </pre>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
