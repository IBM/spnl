import { match, P } from "ts-pattern"

// Sigh, not sure how to get this automatically from Rust, yet. Vite seems to be getting in the way.
export type User = { user: string }
export type System = { system: string }
export type Assistant = { assistant: string }
export type Print = { print: string }
export type Plus = { plus: Query[] }
export type Cross = { cross: Query[] }
export type Repeat = { repeat: { n: number; query: Query } }
export type Generate = {
  g: { model: string; input: Query; max_tokens: number; temperature: number }
}
export type Query =
  | Print
  | User
  | System
  | Assistant
  | Plus
  | Cross
  | Repeat
  | Generate

export function isGenerate(u: Query): u is Generate {
  return match(u)
    .with({ g: P._ }, () => true)
    .otherwise(() => false)
}
