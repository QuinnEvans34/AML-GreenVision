/**
 * Display formatting for raw ImageFolder class names.
 *
 * Per IMPLEMENTATION_GUIDE.md Decision 10. Lives in the frontend so the
 * API stays presentation-agnostic.
 *
 * Examples:
 *   "Apple___healthy"                  -> "Apple (healthy)"
 *   "Tomato___Late_blight"             -> "Tomato — Late blight"
 *   "Pepper,_bell___Bacterial_spot"    -> "Bell pepper — Bacterial spot"
 *   "Background_without_leaves"        -> "No leaf detected"
 */

export interface FormattedClassName {
  crop: string;
  condition: string;
  isHealthy: boolean;
  isBackground: boolean;
}

export function formatClassName(raw: string): FormattedClassName {
  if (raw === "Background_without_leaves") {
    return {
      crop: "",
      condition: "No leaf detected",
      isHealthy: false,
      isBackground: true,
    };
  }

  const [cropRaw, conditionRaw = ""] = raw.split("___");

  // Pepper,_bell -> Bell pepper
  let crop = cropRaw.replace(/_/g, " ").trim();
  if (crop === "Pepper, bell") crop = "Bell pepper";

  if (conditionRaw === "healthy") {
    return { crop, condition: "healthy", isHealthy: true, isBackground: false };
  }

  const condition = conditionRaw.replace(/_/g, " ").trim();
  return { crop, condition, isHealthy: false, isBackground: false };
}

export function classToDisplay(raw: string): string {
  const f = formatClassName(raw);
  if (f.isBackground) return f.condition;
  if (f.isHealthy) return `${f.crop} (healthy)`;
  return `${f.crop} — ${f.condition}`;
}
