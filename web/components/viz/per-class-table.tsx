"use client";

import * as React from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

import type { PerClassMetric } from "@/lib/types";
import { cn } from "@/lib/utils";

interface PerClassTableProps {
  metrics: PerClassMetric[];
}

type SortKey = "display_name" | "precision" | "recall" | "f1" | "support";
type SortDir = "asc" | "desc";

export function PerClassTable({ metrics }: PerClassTableProps) {
  const [query, setQuery] = React.useState("");
  const [sortKey, setSortKey] = React.useState<SortKey>("f1");
  const [sortDir, setSortDir] = React.useState<SortDir>("asc");

  const filtered = React.useMemo(() => {
    const q = query.trim().toLowerCase();
    let rows = metrics;
    if (q) {
      rows = rows.filter(
        (m) =>
          m.display_name.toLowerCase().includes(q) ||
          m.class_name.toLowerCase().includes(q),
      );
    }
    rows = [...rows].sort((a, b) => {
      let aVal: string | number = a[sortKey];
      let bVal: string | number = b[sortKey];
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDir === "asc"
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }
      const diff = (aVal as number) - (bVal as number);
      return sortDir === "asc" ? diff : -diff;
    });
    return rows;
  }, [metrics, query, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "display_name" ? "asc" : "desc");
    }
  }

  return (
    <Card>
      <CardHeader className="space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <CardTitle className="text-base">Per-class metrics</CardTitle>
            <p className="text-xs text-muted-foreground">
              All 39 classes — sort by any column. The lowest-F1 rows are where
              the model's 0.27% errors live.
            </p>
          </div>
          <Input
            placeholder="Filter classes…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full sm:w-64"
          />
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-y bg-muted/30">
              <tr className="text-left">
                <Th label="Class"     columnKey="display_name" activeKey={sortKey} sortDir={sortDir} toggleSort={toggleSort} />
                <Th label="Precision" columnKey="precision"    activeKey={sortKey} sortDir={sortDir} toggleSort={toggleSort} align="right" />
                <Th label="Recall"    columnKey="recall"       activeKey={sortKey} sortDir={sortDir} toggleSort={toggleSort} align="right" />
                <Th label="F1"        columnKey="f1"           activeKey={sortKey} sortDir={sortDir} toggleSort={toggleSort} align="right" />
                <Th label="Support"   columnKey="support"      activeKey={sortKey} sortDir={sortDir} toggleSort={toggleSort} align="right" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((m) => (
                <tr
                  key={m.class_name}
                  className="border-b last:border-0 hover:bg-muted/30"
                >
                  <td className="px-4 py-2">
                    <div className="font-medium">{m.display_name}</div>
                    <div className="text-[10px] font-mono text-muted-foreground">
                      {m.class_name}
                    </div>
                  </td>
                  <NumCell value={m.precision} />
                  <NumCell value={m.recall} />
                  <NumCell value={m.f1} highlight />
                  <td className="px-4 py-2 text-right tabular-nums text-muted-foreground">
                    {m.support}
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={5} className="p-6 text-center text-sm text-muted-foreground">
                    No classes match &ldquo;{query}&rdquo;
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function Th({
  label,
  columnKey,
  activeKey,
  sortDir,
  toggleSort,
  align = "left",
}: {
  label: string;
  columnKey: SortKey;
  activeKey: SortKey;
  sortDir: SortDir;
  toggleSort: (k: SortKey) => void;
  align?: "left" | "right";
}) {
  const isActive = activeKey === columnKey;
  return (
    <th
      className={cn(
        "select-none px-4 py-2 text-xs font-medium text-muted-foreground",
        align === "right" && "text-right",
      )}
    >
      <Button
        variant="ghost"
        size="sm"
        className={cn(
          "-mx-2 h-7 gap-1 px-2 text-xs",
          isActive && "text-foreground",
        )}
        onClick={() => toggleSort(columnKey)}
      >
        {label}
        {!isActive ? (
          <ArrowUpDown className="h-3 w-3 opacity-50" />
        ) : sortDir === "asc" ? (
          <ArrowUp className="h-3 w-3" />
        ) : (
          <ArrowDown className="h-3 w-3" />
        )}
      </Button>
    </th>
  );
}

function NumCell({ value, highlight = false }: { value: number; highlight?: boolean }) {
  const pct = value * 100;
  return (
    <td className="px-4 py-2 text-right">
      <div className="flex items-center justify-end gap-2">
        <span
          className={cn(
            "tabular-nums",
            highlight ? "font-medium" : "text-muted-foreground",
            value < 0.9 && "text-amber-600 dark:text-amber-400",
            value < 0.5 && "text-rose-600 dark:text-rose-400",
          )}
        >
          {pct.toFixed(1)}%
        </span>
        <Progress value={pct} className="h-1 w-12" />
      </div>
    </td>
  );
}
