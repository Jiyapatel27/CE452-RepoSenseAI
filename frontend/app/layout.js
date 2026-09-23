import Shell from "@/components/Shell";

import "./globals.css";

export const metadata = {
  title: "RepoSense AI",
  description: "Ask questions about any GitHub repository, answered from code.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
