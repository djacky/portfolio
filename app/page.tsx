import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import About from "@/components/About";
import Experience from "@/components/Experience";
import RecruiterMatch from "@/components/RecruiterMatch";
import DemoSwitcher from "@/components/DemoSwitcher";
import Skills from "@/components/Skills";
import Publications from "@/components/Publications";
import Contact from "@/components/Contact";

export default function Page() {
  return (
    <main className="relative">
      <Nav />
      <Hero />
      <RecruiterMatch />
      <About />
      <Experience />
      <section id="demos" className="relative mx-auto max-w-6xl px-6 py-24">
        <header className="mb-12 text-center">
          <p className="text-xs uppercase tracking-[0.25em] text-accent2">Live Demos</p>
          <h2 className="mt-2 text-4xl font-semibold text-gradient">Don&apos;t take my word for it</h2>
          <p className="mt-3 text-gray-400 max-w-2xl mx-auto">
            These aren&apos;t screenshots — they&apos;re the actual algorithms, running live in your browser.
            Train a policy, tweak a controller, watch embeddings converge.
          </p>
        </header>
        <DemoSwitcher />
      </section>
      <Skills />
      <Publications />
      <Contact />
      <footer className="py-10 text-center text-xs text-gray-500">
        © {new Date().getFullYear()} Achille Nicoletti — Built with Next.js, Tailwind, Framer Motion. Hosted on Vercel.
      </footer>
    </main>
  );
}
