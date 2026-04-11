import type { Metadata } from "next";
import "./globals.css";
import { ContactDrawerProvider } from "@/components/ContactDrawer";
import { siteFont } from "@/lib/fonts";

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
    <html lang="en" className={siteFont.variable}>
      <body className="font-sans">
        <ContactDrawerProvider>{children}</ContactDrawerProvider>
      </body>
    </html>
  );
}
