/**
 * FIX: Sign In button in the homepage header
 * ------------------------------------------
 * Problem: The button had no onClick and was not a Link, so it did nothing.
 *
 * Your app already has a complete Supabase auth system at /auth
 * (email/password + Google OAuth). The button only needed to navigate there.
 *
 * HOW TO APPLY IN LOVABLE:
 * 1. Open src/routes/index.tsx
 * 2. Find the Header function
 * 3. Replace the broken <button>Sign In</button> with the Link version below
 */

// ========== CORRECTED HEADER SNIPPET ==========
// Make sure Link is already imported from @tanstack/react-router

function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border">
      <div className="mx-auto max-w-6xl px-4 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <img src={logo} alt="PrimalPrint AI logo" width={36} height={36} className="h-9 w-9" />
          <span className="font-script text-2xl">PrimalPrint AI</span>
        </Link>
        <div className="flex items-center gap-3">
          {/* FIXED: Now correctly goes to the existing Supabase auth page */}
          <Link
            to="/auth"
            className="rounded-full border border-border px-4 py-1.5 text-sm hover:bg-muted transition"
          >
            Sign In
          </Link>
          <button onClick={() => setOpen((v) => !v)} aria-label="menu" className="p-2">
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>
      {open ? (
        <div className="mx-auto max-w-6xl px-4 pb-4 flex flex-col gap-3 text-sm">
          <Link to="/create" className="py-2">
            Create Stencil
          </Link>
          <a href="#how" className="py-2">
            How it works
          </a>
          <a href="#results" className="py-2">
            Best Results
          </a>
          <a href="#preview" className="py-2">
            See Examples
          </a>
          <Link to="/vault" className="py-2 text-primary font-semibold">
            Saved Generations / Storage Vault
          </Link>
          <Link to="/gallery" className="py-2">
            Community Gallery
          </Link>
          <Link to="/plugins" className="py-2">
            Plugins
          </Link>
          <Link to="/help" className="py-2 text-primary font-semibold">
            Help & Instructions
          </Link>
          {/* Also add Sign In inside the mobile menu for consistency */}
          <Link to="/auth" className="py-2 font-semibold">
            Sign In
          </Link>
        </div>
      ) : null}
    </header>
  );
}

// ========== HOW SUPABASE IS ALREADY CONNECTED ==========
/*
Your app already has full Supabase Auth:

- src/integrations/supabase/client.ts          → Supabase client
- src/routes/auth.tsx                          → Sign in / Sign up page
  - Email + password
  - Google OAuth
  - Redirects to /vault after successful login

When a user clicks the fixed Sign In button:
1. They go to /auth
2. They sign in with email or Google
3. Supabase creates/returns a session
4. The auth page detects the session and redirects them to /vault

No extra connection work is needed for basic Sign In.
Later you can use the same session to:
- Gate classical vs Gemini generations
- Track free tier usage (10 classical gens)
- Store OpenRouter keys for Master Pro users
- Sync vault across devices
*/
