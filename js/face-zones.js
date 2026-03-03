/**
 * face-zones.js — Anatomical zone mapping for MediaPipe Face Mesh (468 landmarks).
 *
 * Maps every relevant landmark to a rhinoplasty-relevant anatomical zone.
 * Each zone has:
 *   - weight (0-1): controls mesh deformation magnitude
 *   - healingRate: controls temporal resolution speed (very_slow → fast)
 *   - color: visualization color for zone overlay
 *   - isBruiseZone: whether bruising appears here
 *
 * healingRate values match HealingModelJS.ZONE_HEALING_RATES:
 *   very_slow — Nasal tip: 12-18 month resolution
 *   slow      — Supratip, columella: 9-12 months
 *   moderate  — Dorsum, alae, nostrils: 6-9 months (baseline)
 *   fast      — Periorbital, cheeks, lip: 2-6 weeks
 *
 * Landmark indices reference: MediaPipe Face Mesh 468-point topology.
 * See: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
 *
 * DISCLAIMER: Illustrative simulation only. Not medical advice.
 */

const FaceZones = (() => {

    // ═══════════════════════════════════════════════════════════════════════
    // NOSE SUB-REGIONS — High-detail mapping for rhinoplasty simulation
    // ═══════════════════════════════════════════════════════════════════════

    const ZONES = {

        // ── NASAL TIP (Pronasale) ─────────────────────────────────────────
        // The most projecting point of the nose. Maximum swelling zone.
        // Recovery: swelling persists 6-18 months, especially with thick skin.
        nasal_tip: {
            landmarks: [1, 2, 4, 5, 19, 94, 141, 370],
            weight: 1.0,
            healingRate: 'very_slow',
            color: [1.0, 0.2, 0.2],  // red — highest impact
            label: "Nasal Tip (Pronasale)",
            description: "Maximum swelling zone. Longest recovery (up to 18 months for thick skin)."
        },

        // ── SUPRATIP ──────────────────────────────────────────────────────
        // Just above the tip, transition from dorsum to tip.
        // Supratip break is a key aesthetic landmark in rhinoplasty.
        supratip: {
            landmarks: [4, 5, 45, 275, 44, 274, 195],
            weight: 0.9,
            healingRate: 'slow',
            color: [1.0, 0.35, 0.15],
            label: "Supratip",
            description: "Transition zone above tip. Significant swelling, defines tip projection."
        },

        // ── NASAL DORSUM (Bridge) ─────────────────────────────────────────
        // The ridge from nasion (root) to supratip.
        // Osteotomy-related swelling concentrates here.
        nasal_dorsum: {
            landmarks: [6, 122, 168, 193, 195, 197, 351, 417, 245, 465, 196, 419, 248, 281, 51, 3, 196, 248],
            weight: 0.7,
            healingRate: 'moderate',
            color: [1.0, 0.55, 0.0],  // orange
            label: "Nasal Dorsum (Bridge)",
            description: "Moderate swelling from osteotomy. Resolves faster than tip."
        },

        // ── COLUMELLA ─────────────────────────────────────────────────────
        // The strip of tissue between nostrils, below the tip.
        // Open rhinoplasty incision site — significant localized swelling.
        columella: {
            landmarks: [0, 2, 164, 165, 167],
            weight: 0.8,
            healingRate: 'slow',
            color: [1.0, 0.4, 0.1],
            label: "Columella",
            description: "Between nostrils. Open rhinoplasty incision site. Significant swelling."
        },

        // ── LEFT ALAR (Nasal Wing) ────────────────────────────────────────
        // Lateral nasal wall / wing. Lower lateral cartilage zone.
        alar_left: {
            landmarks: [
                48, 49, 50, 51, 52, 53, 54,     // alar contour
                64, 75, 79,                        // alar base
                102, 103, 104, 105, 106,           // alar rim
                174, 188, 196, 198, 209            // alar-cheek junction
            ],
            weight: 0.5,
            healingRate: 'moderate',
            color: [1.0, 0.75, 0.0],  // amber
            label: "Left Alar (Nasal Wing)",
            description: "Moderate swelling. Alar reduction site if performed."
        },

        // ── RIGHT ALAR (Nasal Wing) ───────────────────────────────────────
        alar_right: {
            landmarks: [
                278, 279, 280, 281, 282, 283, 284,  // alar contour
                294, 305, 309,                        // alar base
                331, 332, 333, 334, 335,              // alar rim
                399, 412, 420, 422, 429               // alar-cheek junction
            ],
            weight: 0.5,
            healingRate: 'moderate',
            color: [1.0, 0.75, 0.0],
            label: "Right Alar (Nasal Wing)",
            description: "Moderate swelling. Alar reduction site if performed."
        },

        // ── LEFT NOSTRIL (Naris) ──────────────────────────────────────────
        // Nostril rim and internal border. Swelling affects breathing.
        nostril_left: {
            landmarks: [
                20, 44, 45, 59, 60,
                166, 218, 219, 235, 236,
                237, 238, 239, 240, 241, 242
            ],
            weight: 0.6,
            healingRate: 'moderate',
            color: [0.9, 0.6, 0.1],
            label: "Left Nostril Rim",
            description: "Nostril border. Swelling may temporarily affect airflow."
        },

        // ── RIGHT NOSTRIL (Naris) ─────────────────────────────────────────
        nostril_right: {
            landmarks: [
                250, 274, 275, 289, 290,
                392, 438, 439, 455, 456,
                457, 458, 459, 460, 461, 462
            ],
            weight: 0.6,
            healingRate: 'moderate',
            color: [0.9, 0.6, 0.1],
            label: "Right Nostril Rim",
            description: "Nostril border. Swelling may temporarily affect airflow."
        },

        // ═══════════════════════════════════════════════════════════════════
        // PERIORBITAL ZONES — Bruising dominates here
        // ═══════════════════════════════════════════════════════════════════

        // ── LEFT PERIORBITAL (Infraorbital / Under-eye) ───────────────────
        // Primary bruising zone. Blood pools here after osteotomy.
        periorbital_left: {
            landmarks: [
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                128, 221, 222, 223, 224, 225, 226, 227, 228,
                229, 230, 231, 232, 233, 234
            ],
            weight: 0.4,
            healingRate: 'fast',
            color: [0.5, 0.2, 0.8],  // purple — bruise zone
            label: "Left Periorbital (Under-eye)",
            description: "Primary bruising zone. Ecchymosis peaks Day 2-3, resolves by Day 14.",
            isBruiseZone: true
        },

        // ── RIGHT PERIORBITAL ─────────────────────────────────────────────
        periorbital_right: {
            landmarks: [
                252, 253, 254, 255, 256, 257, 258, 259, 260, 261,
                339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
                357, 441, 442, 443, 444, 445, 446, 447, 448,
                449, 450, 451, 452, 453, 454
            ],
            weight: 0.4,
            healingRate: 'fast',
            color: [0.5, 0.2, 0.8],
            label: "Right Periorbital (Under-eye)",
            description: "Primary bruising zone. Ecchymosis peaks Day 2-3, resolves by Day 14.",
            isBruiseZone: true
        },

        // ═══════════════════════════════════════════════════════════════════
        // CHEEK / MALAR ZONES — Mild swelling, some bruise spread
        // ═══════════════════════════════════════════════════════════════════

        cheek_left: {
            landmarks: [
                35, 36, 47, 100, 101,
                116, 117, 118, 119, 123, 124, 125, 126, 127,
                142, 147, 149, 150,
                169, 170, 171, 175, 176, 177,
                187, 192, 203, 204, 205, 206, 207, 210, 211, 212,
                138, 213, 215
            ],
            weight: 0.2,
            healingRate: 'fast',
            color: [0.3, 0.5, 0.9],  // blue — low impact
            label: "Left Cheek (Malar)",
            description: "Mild swelling. Bruising may spread here from periorbital area.",
            isBruiseZone: true
        },

        cheek_right: {
            landmarks: [
                265, 266, 277, 329, 330,
                345, 346, 347, 348, 352, 353, 354, 355, 356,
                371, 376, 378, 379,
                394, 395, 396, 400, 401, 407,
                411, 416, 423, 424, 425, 426, 427, 430, 431, 432,
                367, 433, 435
            ],
            weight: 0.2,
            healingRate: 'fast',
            color: [0.3, 0.5, 0.9],
            label: "Right Cheek (Malar)",
            description: "Mild swelling. Bruising may spread here from periorbital area.",
            isBruiseZone: true
        },

        // ── UPPER LIP (Cutaneous lip) ─────────────────────────────────────
        // Slight swelling possible if columella work done.
        upper_lip: {
            landmarks: [
                0, 11, 12, 13, 14, 15, 16, 17,
                37, 39, 40, 72, 73, 74,
                165, 167, 164,
                185, 186, 191, 192,
                267, 269, 270, 302, 303, 304,
                393, 409, 410, 415, 416
            ],
            weight: 0.15,
            healingRate: 'fast',
            color: [0.2, 0.6, 0.6],  // teal — minimal
            label: "Upper Lip Area",
            description: "Minimal swelling. Numbness possible if columella incision."
        }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // KEY ANATOMICAL REFERENCE LANDMARKS — for distance-based falloff
    // ═══════════════════════════════════════════════════════════════════════

    const REFERENCE_POINTS = {
        nose_tip:       1,    // Pronasale
        nasion:         6,    // Root of nose / bridge top
        subnasale:      164,  // Base of columella
        left_alar:      48,   // Left wing landmark
        right_alar:     278,  // Right wing landmark
        left_nostril:   60,   // Left nostril rim
        right_nostril:  290,  // Right nostril rim
        left_eye_inner: 133,  // Left medial canthus
        right_eye_inner: 362, // Right medial canthus
        left_infraorb:  116,  // Left infraorbital point
        right_infraorb: 345,  // Right infraorbital point
        left_cheek:     123,  // Left malar prominence
        right_cheek:    352,  // Right malar prominence
        glabella:       168,  // Between eyebrows
        chin:           152,  // Gnathion
    };

    // ═══════════════════════════════════════════════════════════════════════
    // NOSE LANDMARKS COMPLETE LIST — for highlighting in the viewer
    // ═══════════════════════════════════════════════════════════════════════

    // Every landmark that belongs to the nose complex
    const ALL_NOSE_LANDMARKS = new Set([
        // Tip + supratip
        1, 2, 4, 5, 19, 94, 141, 370, 195,
        // Dorsum
        6, 122, 168, 193, 195, 197, 351, 417, 245, 465, 196, 419, 248, 3,
        // Columella
        0, 164, 165, 167,
        // Left alar + nostril
        48, 49, 50, 51, 52, 53, 54, 64, 75, 79,
        102, 103, 104, 105, 106, 174, 188, 198, 209,
        20, 44, 45, 59, 60, 166, 218, 219, 235, 236, 237, 238, 239, 240, 241, 242,
        // Right alar + nostril
        278, 279, 280, 281, 282, 283, 284, 294, 305, 309,
        331, 332, 333, 334, 335, 399, 412, 420, 422, 429,
        250, 274, 275, 289, 290, 392, 438, 439, 455, 456, 457, 458, 459, 460, 461, 462
    ]);

    // ═══════════════════════════════════════════════════════════════════════
    // WEIGHT COMPUTATION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Build a lookup: landmarkIndex -> { zoneName, weight, color, isBruiseZone, healingRate }
     */
    function buildLandmarkMap() {
        const map = new Map();
        for (const [zoneName, zone] of Object.entries(ZONES)) {
            for (const idx of zone.landmarks) {
                // If already assigned, keep the higher weight (more specific zone wins)
                const existing = map.get(idx);
                if (!existing || zone.weight > existing.weight) {
                    map.set(idx, {
                        zone: zoneName,
                        weight: zone.weight,
                        color: zone.color,
                        isBruiseZone: zone.isBruiseZone || false,
                        healingRate: zone.healingRate || 'moderate'
                    });
                }
            }
        }
        return map;
    }

    // ── Healing rate constants for weighted blending ──
    const HEALING_RATE_ORDER = ['very_slow', 'slow', 'moderate', 'fast'];

    /**
     * Compute per-vertex zone weights for all landmarks.
     * Uses a hybrid approach:
     * 1) Hard assignment for landmarks explicitly listed in zones
     * 2) Distance-based gaussian falloff with WEIGHTED ACCUMULATION for unlisted landmarks
     *
     * Key improvements over winner-take-all:
     *   - Accumulates ALL reference influences (smooth transitions between zones)
     *   - Outputs continuous bruiseBlend (0-1 float) instead of boolean
     *   - Outputs healingRateWeights for smooth swelling blending across zone boundaries
     *
     * @param {Array} landmarks - Array of {x, y, z} positions
     * @returns {Array} - Array of {weight, color, isBruiseZone, bruiseBlend, zone, healingRate, healingRateWeights}
     */
    function computeZoneWeights(landmarks) {
        const landmarkMap = buildLandmarkMap();
        const N = landmarks ? landmarks.length : 468;
        const weights = new Array(N);

        // First pass: assign explicit zone members with full bruiseBlend
        for (let i = 0; i < N; i++) {
            const explicit = landmarkMap.get(i);
            if (explicit) {
                // Compute bruiseBlend for explicit zone members
                let bruiseBlend = 0;
                if (explicit.isBruiseZone) {
                    bruiseBlend = explicit.weight * 2.0; // amplified for bruise zones
                } else if (explicit.zone.startsWith("nasal_") || explicit.zone === "supratip") {
                    bruiseBlend = explicit.weight * 0.3; // mild for nose zones
                }
                // Build healingRateWeights: 100% for the assigned rate
                const hrWeights = { very_slow: 0, slow: 0, moderate: 0, fast: 0 };
                hrWeights[explicit.healingRate || 'moderate'] = 1.0;

                weights[i] = {
                    ...explicit,
                    bruiseBlend: Math.min(1, bruiseBlend),
                    healingRateWeights: hrWeights,
                };
            } else {
                weights[i] = null; // will be computed via distance
            }
        }

        // Second pass: WEIGHTED ACCUMULATION for unassigned vertices
        // Instead of winner-take-all, blend all nearby reference influences
        for (let i = 0; i < N; i++) {
            if (weights[i] !== null) continue;

            const pos = landmarks[i];
            if (!pos || typeof pos.x !== 'number') {
                weights[i] = {
                    zone: "none", weight: 0, color: [0.15, 0.15, 0.15],
                    isBruiseZone: false, bruiseBlend: 0,
                    healingRate: 'moderate',
                    healingRateWeights: { very_slow: 0, slow: 0, moderate: 1, fast: 0 },
                };
                continue;
            }

            // Accumulate all influences
            let totalInfluence = 0;
            let accR = 0, accG = 0, accB = 0;
            let accBruise = 0;
            let bestZone = "none";
            let bestInfluence = 0;
            const hrAccum = { very_slow: 0, slow: 0, moderate: 0, fast: 0 };

            for (const [refName, refIdx] of Object.entries(REFERENCE_POINTS)) {
                const refPos = landmarks[refIdx];
                if (!refPos) continue;

                const refZone = landmarkMap.get(refIdx);
                if (!refZone) continue;

                const dx = pos.x - refPos.x;
                const dy = pos.y - refPos.y;
                const dz = pos.z - refPos.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

                // Wider Gaussian falloff for smooth transitions
                const sigma = 0.12; // ~12% of face width (~17mm)
                const influence = refZone.weight * Math.exp(-(dist * dist) / (2 * sigma * sigma));

                if (influence < 0.001) continue; // skip negligible

                totalInfluence += influence;
                const c = Array.isArray(refZone.color) ? refZone.color : [0.15, 0.15, 0.15];
                accR += c[0] * influence;
                accG += c[1] * influence;
                accB += c[2] * influence;

                // Accumulate bruise blend
                let refBruise = 0;
                if (refZone.isBruiseZone) refBruise = 2.0;
                else if (refZone.zone.startsWith("nasal_") || refZone.zone === "supratip") refBruise = 0.3;
                accBruise += refBruise * influence;

                // Accumulate healing rate weights
                const hr = refZone.healingRate || 'moderate';
                hrAccum[hr] = (hrAccum[hr] || 0) + influence;

                // Track dominant zone for display purposes
                if (influence > bestInfluence) {
                    bestInfluence = influence;
                    bestZone = refZone.zone;
                }
            }

            if (totalInfluence > 0.001) {
                // Normalize
                const invTotal = 1.0 / totalInfluence;
                const blendedColor = [accR * invTotal, accG * invTotal, accB * invTotal];
                const blendedBruise = Math.min(1, accBruise * invTotal);

                // Normalize healing rate weights
                const hrTotal = hrAccum.very_slow + hrAccum.slow + hrAccum.moderate + hrAccum.fast;
                if (hrTotal > 0) {
                    for (const k of HEALING_RATE_ORDER) hrAccum[k] /= hrTotal;
                }

                // Dominant healing rate = highest weight
                let bestHR = 'moderate';
                let bestHRW = 0;
                for (const k of HEALING_RATE_ORDER) {
                    if (hrAccum[k] > bestHRW) { bestHRW = hrAccum[k]; bestHR = k; }
                }

                weights[i] = {
                    zone: bestZone,
                    weight: Math.max(0, Math.min(1, bestInfluence)),
                    color: blendedColor,
                    isBruiseZone: blendedBruise > 0.3,
                    bruiseBlend: blendedBruise,
                    healingRate: bestHR,
                    healingRateWeights: { ...hrAccum },
                };
            } else {
                weights[i] = {
                    zone: "none", weight: 0, color: [0.15, 0.15, 0.15],
                    isBruiseZone: false, bruiseBlend: 0,
                    healingRate: 'moderate',
                    healingRateWeights: { very_slow: 0, slow: 0, moderate: 1, fast: 0 },
                };
            }
        }

        return weights;
    }

    /**
     * Get the deformation weight for a vertex given its zone assignment.
     * Swelling zones: weight controls mesh displacement magnitude.
     * Bruise zones: weight controls color overlay intensity.
     */
    function getSwellingWeight(zoneData) {
        if (!zoneData) return 0;
        // Bruise-only zones still get some swelling
        return zoneData.weight;
    }

    function getBruisingWeight(zoneData) {
        if (!zoneData) return 0;
        // Use continuous bruiseBlend for smooth transitions between bruise/non-bruise zones
        if (typeof zoneData.bruiseBlend === 'number') {
            return zoneData.bruiseBlend * (zoneData.weight || 0);
        }
        // Legacy fallback for non-blended data
        if (zoneData.isBruiseZone) return zoneData.weight * 2.0;
        if (zoneData.zone && (zoneData.zone.startsWith("nasal_") || zoneData.zone === "supratip")) return zoneData.weight * 0.3;
        return 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PUBLIC API
    // ═══════════════════════════════════════════════════════════════════════

    return {
        ZONES,
        REFERENCE_POINTS,
        ALL_NOSE_LANDMARKS,
        buildLandmarkMap,
        computeZoneWeights,
        getSwellingWeight,
        getBruisingWeight,
    };

})();

// Export for non-module usage
if (typeof window !== 'undefined') {
    window.FaceZones = FaceZones;
}
