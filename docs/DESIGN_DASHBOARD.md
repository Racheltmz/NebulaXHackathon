# Midnight Banking Dashboard

## Overview

Midnight Banking is a dark-first design system for personal banking and fintech dashboards. Its foundation of deep midnight navy creates a calm, premium canvas against which gradient account cards and warm financial-product cards become the visual focal points. Status colors lean confident: a soft mint green for incoming amounts, a coral red for outgoing. The system prioritizes scannability of monetary values, clear separation between primary actions and informational content, and a polished, modern aesthetic appropriate for high-trust financial flows.

Source: extracted from a Figma "Home Banking" desktop dashboard (1600 × 1056 canvas), tokenized into a reusable system.

---

## Colors

### Page & Surfaces
- **Page Background** (#040D3F): Very dark midnight navy — primary page canvas
- **Surface Default** (#141C4E): Slightly lighter navy — header bars, large surface regions
- **Card Background** (#1D2552): Card and tile fill — transaction rows, KPI tiles, sidebar items
- **Card Background Hover** (#243066): Hover state for cards and rows

### Foreground / Text
- **Text Primary** (#FFFFFF): Headlines, primary labels, on-card text
- **Text Secondary** (#B9BBC7): Muted body, subtitles, "View All" links
- **Text Tertiary** (#6A7178): Disabled states, very low-emphasis text
- **Neutral 4** (#DEE2E6): Light dividers
- **Neutral 6** (#ADB5BD): Neutral badge fills, mid-grey text on light surfaces

### Status / Signed Amounts
- **Positive** (#7FE089): Incoming transactions, positive deltas, success badges
- **Negative** (#F97D7D): Outgoing transactions, negative deltas, error states
- **Info Cyan** (#0D8091): Cyan accent — informational badges, secondary CTAs

### Account Card Gradients
Three named gradients that drive the horizontal account-card carousel. Each is a 135° linear gradient from a deeper anchor stop to a lighter accent.
- **Card Turquoise**: linear-gradient(135deg, #0D8091 0%, #1AB8C7 100%) — Checking accounts
- **Card Pink**: linear-gradient(135deg, #B53D7F 0%, #E45D9A 100%) — Savings accounts
- **Card Purple**: linear-gradient(135deg, #5A2D8A 0%, #8B4FBF 100%) — Credit / charge accounts

### Financial-Product Card Gradients
Used for sidebar promotional cards (loans, retirement, goal CTAs). Higher-saturation, marketing-tone.
- **Product Yellow**: linear-gradient(135deg, #D4A017 0%, #F2C94C 100%) — Personal Loan, Credit Card Request, Retirement Plan
- **Product Pink Soft**: linear-gradient(135deg, #E64980 0%, #FF6B9D 100%) — "Define New Goal" CTAs
- **Product Cyan Soft**: linear-gradient(135deg, #2D7DD2 0%, #4FA8E8 100%) — "For You" recommendations, customised solutions

---

## Typography

- **Headline Font**: Inter
- **Body Font**: Inter
- **Mono Font**: JetBrains Mono (for tabular monetary alignment)

- **Display**: Inter 32px bold, 1.2 line height — page-level totals, hero balance values
- **H1 / Page Title**: Inter 22px bold, 1.25 line height — KPI primary values ("$3,689.12"), card balance numbers
- **H2 / Section Title**: Inter 16px semibold, 1.3 line height — "Total Balance", "Last Transactions", "Your Goals", "Financial Assets", "For you"
- **Card Title**: Inter 16px semibold, 1.3 line height — gradient-card account names, goal/product card titles
- **Body**: Inter 14px medium, 1.4 line height — transaction merchant names, nav links, action button labels, hero greeting
- **Body Subtle**: Inter 13px medium, 1.4 line height — amount lines ("$543 of $1,000"), "View All" links
- **Caption**: Inter 12px regular, 1.4 line height — transaction subtitles, KPI labels, dates, delta percentages
- **Micro**: Inter 11px semibold, 1.3 line height — status badges ("Under Approval"), small chips
- **Mono Numeric**: JetBrains Mono 14px medium, 1.4 line height — tabular amounts, account numbers, last-4 digits

---

## Spacing

Base unit: **8px**
- **xs**: 4px — Inline icon gaps, badge padding
- **sm**: 8px — Tight component padding, gap between adjacent rows
- **md**: 12px — Pill button gap, action-row spacing
- **lg**: 16px — Default card/tile padding, gap between cards in carousel
- **xl**: 20px — Inside the gradient cards
- **2xl**: 24px — Outer page padding, gap between main column and sidebar, gap between sidebar sections
- **3xl**: 32px — Major section breaks within a column

---

## Border Radius

- **xs** (6px): Status badges, small chips ("Under Approval")
- **sm** (8px): Light dividers, small accents
- **DEFAULT** (12px): Tile and row backgrounds — transaction rows, KPI tiles, sidebar product cards, goal cards
- **lg** (16px): Large gradient account cards in the carousel
- **full** (9999px): Pill buttons (Transfer / Pay / Add / View / Account Insights), avatar circles, progress-bar tracks

---

## Elevation

Banking dashboards use a two-tier shadow system: a tight subtle shadow for tiles, and a deep ambient glow for hero cards.

- **Tile Shadow** (`0 2px 4px 0 rgba(0, 0, 0, 0.10)`): Transaction rows, KPI tiles, sidebar items, dropdown menus
- **Card Shadow Big** (`0 0 60px 0 rgba(0, 4, 26, 0.60)`): Gradient account cards in the carousel — produces a soft midnight glow against the page background
- **Modal Shadow** (`0 8px 32px 0 rgba(0, 0, 0, 0.30)`): Modals, bottom sheets

---

## Components

### Header Bar
- **Height**: 88px, full page width
- **Background**: `--surface-default` (#141C4E)
- **Layout** (left → right): Wordmark logo, centered nav (5 links, 24px gap), right cluster (icon trio + welcome greeting + avatar)
- **Wordmark**: Inter 20px bold, `--text-primary`
- **Nav links**: Inter 14px medium, `--text-secondary` (default) / `--text-primary` underlined (active)
- **Right icons**: 20×20, `--text-secondary`, hoverable to `--text-primary`
- **Avatar**: 40×40 circle, background `--card-bg`, initials in `--text-primary` 14px semibold

### Gradient Account Card
- **Size**: ~340px wide × 180px tall
- **Background**: One of `--card-turquoise`, `--card-pink`, `--card-purple`
- **Padding**: 20px
- **Radius**: 16px
- **Shadow**: Card Shadow Big (deep ambient midnight glow)
- **Top row**: Card title left (Inter 16px semibold, white), small network wordmark / chip top-right (white 80% alpha)
- **Bottom row**: Masked number left ("•••• 0001", JetBrains Mono 14px medium, white 80% alpha), balance value top-right (Inter 22px bold, white)
- **Carousel layout**: 3 cards horizontal, 16px gap, no scrollbar on desktop

### Pill Action Button
- **Padding**: 8px 16px (compact) / 10px 20px (default)
- **Radius**: full (9999px)
- **Font**: Inter 13px semibold
- **Variants**:
  - **Primary on dark**: White (#FFFFFF) fill, `--page-bg` (#040D3F) text — used for "Transfer / Pay / Add" action row
  - **Toggle Active**: `--card-bg` (#1D2552) fill, `--text-primary` text — used for active toggle state ("Show Balance")
  - **Toggle Inactive**: transparent fill, `--text-secondary` text, 1px `--card-bg` border
  - **Small View**: White fill, `--page-bg` text, smaller padding (6px 12px) — used inside product cards as "View" / "Add"

### Transaction Row
- **Background**: `--card-bg` (#1D2552)
- **Padding**: 16px
- **Radius**: 12px
- **Margin between rows**: 8px
- **Layout** (left → right):
  - 32×32 circular icon container — green tint (`#7FE08920`) for Incoming, red tint (`#F97D7D20`) for Outgoing — with `ArrowUp` / `ArrowDown` icon centered
  - Stacked center text — DisplayType (Inter 14px semibold, white) + MerchantName (Inter 12px regular, `--text-secondary`)
  - Stacked right text — signed amount (Inter 14px semibold, color from direction) + date (Inter 12px regular, `--text-secondary`, "dd MMM" format)

### KPI Tile
- **Background**: `--card-bg` (#1D2552)
- **Padding**: 16px
- **Radius**: 12px
- **Gap between tiles**: 8px
- **Anatomy**:
  - Small icon top-left (20×20, `--text-secondary`)
  - Label below icon — Inter 12px medium, `--text-secondary`
  - Primary value — Inter 22px bold, `--text-primary`
  - Delta line — Inter 12px semibold, `--positive` or `--negative` (with leading "+" / "-")

### Goal / Product Card
- **Background**: `--card-bg` (default goal card) or one of the product gradients (loan/promo cards)
- **Padding**: 16px
- **Radius**: 12px
- **Min-height**: 80px (compact) / unbounded (full)
- **Anatomy**:
  - Top row: small icon left (20×20), title center (Inter 14px semibold, white), action pill right ("Add" / "View" — small variant)
  - Body line: amount or subtitle (Inter 13px medium, `--text-secondary` on `--card-bg`, white-80% on gradient cards)
  - Optional progress bar (full-width, 4px tall — see Progress Bar)
  - Optional decorative illustration on the right at white 30% alpha

### Progress Bar
- **Height**: 4px
- **Track background**: `--card-bg-hover` (#243066) on dark cards, white-30% on gradient cards
- **Fill color**: `--positive` (#7FE089) for goals on track, `--text-primary` (white) on gradient product cards
- **Radius**: full (rounded ends)

### Status Badge
- **Padding**: 4px 8px
- **Radius**: 6px
- **Font**: Inter 11px semibold
- **Variants**:
  - **Neutral**: `--neutral-badge` (#ADB5BD) fill, `--text-primary` text — for "Under Approval", "Pending"
  - **Success**: `--positive` 15% fill, `--positive` text — for "Active", "On Track"
  - **Error**: `--negative` 15% fill, `--negative` text — for "Behind", "Failed"
  - **Info**: `--info-cyan` 15% fill, `--info-cyan` text — for "New", "Achieved"

### Avatar
- **Size**: 40×40 (default) / 32×32 (compact)
- **Radius**: full (circle)
- **Background**: `--card-bg` (#1D2552)
- **Content**: Initials (1–2 chars) centered in Inter 14px semibold, `--text-primary`

### Section Header
- **Layout**: Title left, "View All" link right
- **Title**: Inter 16px semibold, `--text-primary`
- **Link**: Inter 13px medium, `--text-secondary`, underline on hover
- **Spacing**: 16px below the header before the content begins

---

## Do's and Don'ts

1. **Do** keep the page background as the deepest tone — surfaces and cards always step lighter, never darker.
2. **Do** reserve gradient cards for primary financial entities (accounts, loans, goal CTAs). Don't use gradients for transactional rows or list items — they lose emphasis when overused.
3. **Don't** mix more than 3 different gradients in a single viewport. The vibrant gradients are accent moments, not wallpaper.
4. **Do** color-code monetary amounts: green for inbound, red for outbound. Apply the color to both the amount text *and* the icon container's tinted background.
5. **Don't** use pure white (#FFFFFF) for body text against the dark backgrounds — soften to `--text-secondary` (#B9BBC7) for non-primary content; reserve pure white for headings, primary actions, and on-card titles.
6. **Do** use JetBrains Mono (or any tabular-numeral font) for amount columns and account numbers. Misaligned digits in a banking dashboard are a credibility leak.
7. **Don't** add hard borders between dark surfaces. Separation comes from background-color steps (#040D3F → #141C4E → #1D2552), not from strokes.
8. **Do** lean on the deep ambient glow (Card Shadow Big) only on the gradient hero cards. On every other tile, use the gentle `Tile Shadow`.
9. **Don't** use small radii (≤4px) anywhere. The system reads as "premium fintech" because of consistent generous rounding (12px tiles, 16px cards, full-pill buttons).
10. **Do** ensure every interactive element has a hover state — even on dark backgrounds. Brightening the surface (`--card-bg-hover`) is enough; avoid color-shifting hover states that change the meaning of the element.
11. **Do** set 24px as the default outer page padding. Tighter feels cramped; looser breaks the grid against the right sidebar.
12. **Don't** introduce neon or saturated accent colors outside the palette. The system's premium feel depends on disciplined use of muted greens and corals against deep navy.
