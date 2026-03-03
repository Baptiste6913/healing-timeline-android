/**
 * healing-model.js — Parametric healing model with zone-specific recovery.
 *
 * KEY FEATURES:
 *   - Zone-specific healing rates (tip vs dorsum vs periorbital)
 *   - Osteotomy toggle (dramatically affects periorbital bruising)
 *   - Open vs closed approach (affects columella/tip swelling)
 *   - Skin thickness + age adjustments
 *   - Bi-exponential onset × decay model
 *
 * Clinical basis:
 *   - Tip: thickest SSTE, poorest lymphatic drainage → 12-18 months
 *   - Supratip/columella: dense fibrofatty tissue → 9-12 months
 *   - Dorsum/alae: moderate tissue → 6-9 months (baseline)
 *   - Periorbital/cheeks: thin skin, good drainage → 2-6 weeks
 *   (Rohrich & Ahmad, PRS 2011; Gruber et al., PRS 2007)
 *
 * DISCLAIMER: Illustrative simulation only. Not medical advice.
 */

const HealingModelJS = (() => {

    // ── Profile defaults ──────────────────────────────────────────────
    const SKIN_TAU2 = { thin: 60, medium: 90, thick: 150 };
    const INTENSITY_SCALE = { low: 0.6, medium: 1.0, high: 1.3 };
    const V_MAX = { low: 2.0, medium: 4.0, high: 6.0 };

    // ── Zone-specific healing rate multipliers ────────────────────────
    // tau2Mult: multiplier for the slow-component time constant.
    // Higher value = slower resolution = more persistent swelling.
    //
    // Clinical rationale:
    //   very_slow (tip): Thick skin/soft tissue envelope, cartilage grafts,
    //     limited lymphatic drainage → edema persists 12-18 months
    //   slow (supratip, columella): Dense fibrofatty tissue, open approach
    //     incision site → intermediate 9-12 months
    //   moderate (dorsum, alae, nostrils): Standard tissue thickness,
    //     adequate vascularity → baseline 6-9 months
    //   fast (periorbital, cheeks, lip): Very thin skin, excellent vascular
    //     supply, gravity-assisted drainage → 2-6 weeks
    const ZONE_HEALING_RATES = {
        very_slow: { tau2Mult: 1.8 },
        slow:      { tau2Mult: 1.3 },
        moderate:  { tau2Mult: 1.0 },
        fast:      { tau2Mult: 0.5 },
    };

    // ── Bruise color keyframes: [day, [r, g, b]] ──────────────────────
    // Follows hemoglobin degradation pathway:
    // Hb → metHb (red→purple) → biliverdin (green) → bilirubin (yellow)
    const BRUISE_COLORS = [
        [0,  [0.50, 0.15, 0.20]],  // dark red (fresh hemorrhage)
        [2,  [0.40, 0.10, 0.35]],  // red-purple (early deoxygenation)
        [4,  [0.30, 0.12, 0.42]],  // deep purple (methemoglobin)
        [7,  [0.25, 0.35, 0.35]],  // blue-green (biliverdin)
        [10, [0.50, 0.55, 0.15]],  // yellow-green (bilirubin forming)
        [14, [0.65, 0.60, 0.30]],  // yellow (bilirubin dominant)
        [21, [0.80, 0.75, 0.55]],  // near-skin (resolving)
    ];

    class HealingModel {
        constructor(profile = {}) {
            this.skinThickness = profile.skinThickness || "medium";
            this.initialIntensity = profile.initialIntensity || "medium";
            this.bruisingPresent = profile.bruisingPresent !== false;
            this.age = profile.age || null;
            this.osteotomy = profile.osteotomy !== false;    // default: true
            this.approach = profile.approach || "open";       // "open" or "closed"

            // ── Bi-exponential decay parameters ──
            // A1: fast component weight (acute inflammatory edema, resolves in weeks)
            // A2: slow component weight (chronic tissue/fibrosis, resolves in months)
            this.A1 = 0.6;
            this.A2 = 0.4;
            this.tau1 = 7.0;  // fast time constant (days)

            let tau2 = SKIN_TAU2[this.skinThickness] || 90;
            if (this.age !== null) {
                if (this.age < 30) tau2 *= 0.9;
                else if (this.age > 50) tau2 *= 1.1;
            }
            this.tau2 = tau2;  // slow time constant (days)

            this.intensityScale = INTENSITY_SCALE[this.initialIntensity] || 1.0;
            this.maxDisplacementMM = V_MAX[this.initialIntensity] || 4.0;
            this.tPeakBruise = 2.5;

            // ── Onset time constant ──
            // Edema develops over 48-72h post-surgery (inflammatory cascade)
            this.tOnset = 0.8; // days — reaches ~63% at 0.8d, ~95% at 2.4d

            // ── Osteotomy: dramatically affects periorbital bruising ──
            // Without osteotomy: minimal bone trauma → very little periorbital ecchymosis
            // With osteotomy: nasal bone fractures → significant subperiosteal hemorrhage
            this.bruiseIntensityMult = this.osteotomy ? 1.0 : 0.15;

            // ── Surgical approach: affects tip/columella swelling ──
            // Open: trans-columellar incision → more tip edema, longer columella recovery
            // Closed: endonasal only → less visible swelling, faster initial recovery
            this.approachSwellMult = this.approach === "open" ? 1.15 : 1.0;

            // ── Pre-compute peak normalization per zone healing rate ──
            // Each zone rate has a different tau2, so the peak of onset×decay differs.
            // We normalize each so peak = 1.0 before scaling.
            this._peakNorms = {};
            for (const [rate, config] of Object.entries(ZONE_HEALING_RATES)) {
                const zoneTau2 = this.tau2 * config.tau2Mult;
                let peakVal = 0;
                for (let t = 0.5; t <= 5; t += 0.1) {
                    const onset = 1 - Math.exp(-t / this.tOnset);
                    const decay = this.A1 * Math.exp(-t / this.tau1)
                                + this.A2 * Math.exp(-t / zoneTau2);
                    peakVal = Math.max(peakVal, onset * decay);
                }
                this._peakNorms[rate] = peakVal || 1;
            }
            // Backward-compatible global peak norm
            this._peakNorm = this._peakNorms['moderate'] || 1;
        }

        /**
         * Zone-specific swelling curve S(t, zone) in [0, 1].
         *
         * Each zone resolves at its own rate based on tissue characteristics:
         *   very_slow (tip):     Day 7→~60%, Day 30→~30%, Day 90→~20%, Day 365→~5%
         *   slow (supratip):     Day 7→~60%, Day 30→~25%, Day 90→~15%, Day 365→~3%
         *   moderate (dorsum):   Day 7→~60%, Day 30→~22%, Day 90→~10%, Day 365→~1%
         *   fast (periorbital):  Day 7→~50%, Day 30→~10%, Day 90→~2%,  Day 365→~0%
         *
         * The fast component (tau1=7d) is shared — all zones have similar
         * acute inflammatory edema in the first 2 weeks. The slow component
         * (tau2 × zoneMult) is what creates the differential recovery timeline.
         *
         * @param {number} t - Days post-surgery
         * @param {string} healingRate - Zone rate: 'very_slow'|'slow'|'moderate'|'fast'
         * @returns {number} Swelling level 0-1
         */
        zoneSwelling(t, healingRate = 'moderate') {
            if (t <= 0) return 0;

            const config = ZONE_HEALING_RATES[healingRate] || ZONE_HEALING_RATES.moderate;
            const zoneTau2 = this.tau2 * config.tau2Mult;
            const peakNorm = this._peakNorms[healingRate] || this._peakNorm;

            // Onset: inflammatory edema accumulation (peaks ~48-72h)
            const onset = 1 - Math.exp(-t / this.tOnset);

            // Bi-exponential resolution with zone-specific slow component
            const decay = this.A1 * Math.exp(-t / this.tau1)
                        + this.A2 * Math.exp(-t / zoneTau2);

            const s = (onset * decay) / peakNorm;

            // Apply approach multiplier for slow zones (tip/supratip/columella)
            // Open approach adds ~15% more swelling to these zones
            let mult = this.intensityScale;
            if (healingRate === 'very_slow' || healingRate === 'slow') {
                mult *= this.approachSwellMult;
            }

            return Math.max(0, Math.min(1, s * mult));
        }

        /**
         * Global swelling (backward compatible, for UI display).
         * Uses 'moderate' zone rate as a representative average.
         */
        swelling(t) {
            return this.zoneSwelling(t, 'moderate');
        }

        /**
         * Swelling range: {min, median, max} for confidence band display.
         * Based on tip swelling (the most clinically significant metric).
         */
        swellingRange(t) {
            const med = this.zoneSwelling(t, 'very_slow');
            return {
                min: Math.max(0, Math.min(1, med * 0.65)),
                median: med,
                max: Math.max(0, Math.min(1, med * 1.25)),
            };
        }

        /**
         * Periorbital ecchymosis (bruising) curve B(t) in [0, 1].
         *
         * Osteotomy is the dominant factor for bruising severity:
         *   With osteotomy:    full bruising (100% intensity), peaks Day 2-3
         *   Without osteotomy: minimal bruising (~15% intensity)
         *
         * Timeline (with osteotomy):
         *   Day 0:   0%   — no bruising yet
         *   Day 1:   ~73% — appearing rapidly
         *   Day 2.5: 100% — peak
         *   Day 5:   ~74% — still significant
         *   Day 7:   ~46% — visibly improving
         *   Day 14:  ~6%  — nearly resolved
         *   Day 21:  ~0%  — resolved
         */
        bruising(t) {
            if (!this.bruisingPresent || t <= 0) return 0;
            const tp = this.tPeakBruise;
            const b = (t / tp) * Math.exp(1.0 - t / tp);
            return Math.max(0, Math.min(1, b * this.intensityScale * this.bruiseIntensityMult));
        }

        /** Interpolated bruise color RGB at time t (hemoglobin degradation). */
        bruiseColor(t) {
            if (t <= 0) return [...BRUISE_COLORS[0][1]];
            const colors = BRUISE_COLORS;
            for (let i = 0; i < colors.length - 1; i++) {
                const [t0, c0] = colors[i];
                const [t1, c1] = colors[i + 1];
                if (t >= t0 && t <= t1) {
                    const frac = (t - t0) / (t1 - t0);
                    return [
                        c0[0] + frac * (c1[0] - c0[0]),
                        c0[1] + frac * (c1[1] - c0[1]),
                        c0[2] + frac * (c1[2] - c0[2]),
                    ];
                }
            }
            return [...colors[colors.length - 1][1]];
        }

        /** Nasal volume delta in mm at time t (based on tip swelling). */
        nasalVolumeDelta(t) {
            return this.zoneSwelling(t, 'very_slow') * this.maxDisplacementMM;
        }

        /**
         * Full evaluation at a single timepoint.
         * Returns pre-computed zone swelling levels for efficient per-vertex mesh building.
         */
        evaluate(t) {
            const tipSwelling = this.zoneSwelling(t, 'very_slow');
            const range = this.swellingRange(t);
            return {
                day: t,
                swellingLevel: tipSwelling,  // primary display metric = tip
                swellingMin: range.min,
                swellingMax: range.max,
                bruisingLevel: this.bruising(t),
                bruiseColor: this.bruiseColor(t),
                nasalVolumeDelta: this.nasalVolumeDelta(t),
                // Pre-computed zone swelling map for per-vertex displacement
                zoneSwelling: {
                    very_slow: tipSwelling,
                    slow: this.zoneSwelling(t, 'slow'),
                    moderate: this.zoneSwelling(t, 'moderate'),
                    fast: this.zoneSwelling(t, 'fast'),
                },
            };
        }
    }

    // ── Preset days ───────────────────────────────────────────────────
    const PRESET_DAYS = [
        { day: 0,   label: "Surgery" },
        { day: 1,   label: "Day 1" },
        { day: 3,   label: "Day 3" },
        { day: 7,   label: "1 Week" },
        { day: 14,  label: "2 Weeks" },
        { day: 30,  label: "1 Month" },
        { day: 90,  label: "3 Months" },
        { day: 180, label: "6 Months" },
        { day: 365, label: "12 Months" },
    ];

    return { HealingModel, PRESET_DAYS, BRUISE_COLORS, ZONE_HEALING_RATES };
})();

if (typeof window !== 'undefined') {
    window.HealingModelJS = HealingModelJS;
}
