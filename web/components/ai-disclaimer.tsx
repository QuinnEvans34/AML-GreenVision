import { Info } from "lucide-react";

import { cn } from "@/lib/utils";

interface AIDisclaimerProps {
  strengthened?: boolean;
  className?: string;
}

export function AIDisclaimer({ strengthened = false, className }: AIDisclaimerProps) {
  return (
    <div
      className={cn(
        "flex gap-2 rounded-md p-3 text-xs",
        strengthened
          ? "border border-rose-500/20 bg-rose-500/5 text-rose-700 dark:text-rose-300"
          : "bg-muted text-muted-foreground",
        className,
      )}
    >
      <Info className="mt-0.5 h-4 w-4 shrink-0" />
      <div>
        {strengthened ? (
          <p>
            <strong>Model confidence is low.</strong> This prediction should
            not be relied on alone. Please retake the photo or consult a
            qualified agronomist or your local agricultural extension office.
          </p>
        ) : (
          <p>
            AI-generated diagnosis for educational purposes. For critical
            agricultural decisions, consult a qualified agronomist or your
            local agricultural extension office.
          </p>
        )}
      </div>
    </div>
  );
}
