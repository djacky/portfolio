import type { Metadata } from "next";
import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ContactDrawerProvider } from "@/components/ContactDrawer";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Achille Nicoletti — Senior AI/ML Engineer",
  description:
    "Portfolio of Achille Nicoletti — Senior AI/ML Engineer. PhD EPFL. CERN, Eaton, Disruptive Labs. Python · PyTorch · AWS · C++.",
  openGraph: {
    title: "Achille Nicoletti — Senior AI/ML Engineer",
    description:
      "PhD EPFL. CERN · Eaton · Disruptive Labs. Python, PyTorch, AWS, C++. Interactive portfolio with live ML demos.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="font-body">
        <ContactDrawerProvider>{children}</ContactDrawerProvider>
      </body>
    </html>
  );
}
