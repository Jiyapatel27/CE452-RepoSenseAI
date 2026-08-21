import "./globals.css";

export const metadata = {
  title: "RepoSense AI",
  description: "Ask questions about any GitHub repository.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
