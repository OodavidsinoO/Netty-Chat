import type { Metadata } from "next";
// import { Inter } from "next/font/google";
import "./globals.css";
import { ReactNode } from "react";
import { Suspense } from "react";

// const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Netty Copilot",
  description:
    "Netty: Your computer network co-pilot. An interactive, intelligent assistant tailored for computer network education.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Suspense fallback={null}>{children}</Suspense>
      </body>
    </html>
  );
}
