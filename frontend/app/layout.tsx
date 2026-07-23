import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "migrantBuddy",
  description: "Answers to Singapore employment questions for migrant workers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
