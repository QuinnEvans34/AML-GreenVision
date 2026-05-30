import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import { SiteNav } from "@/components/site-nav";
import { Toaster } from "@/components/ui/sonner";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "GreenVision — Plant Disease Diagnostics",
  description:
    "Upload a leaf photo. Get a diagnosis and treatment recommendation. Powered by EfficientNet-B0 fine-tuned on PlantVillage.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-background text-foreground">
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex min-h-screen flex-col">
            <SiteNav />
            <main className="container mx-auto max-w-6xl flex-1 px-4 py-8">
              {children}
            </main>
            <footer className="border-t py-6">
              <div className="container mx-auto max-w-6xl px-4 text-center text-xs text-muted-foreground">
                GreenVision · EfficientNet-B0 fine-tuned on PlantVillage · Built for AML W10A1
              </div>
            </footer>
          </div>
          <Toaster richColors closeButton position="top-right" />
        </ThemeProvider>
      </body>
    </html>
  );
}
