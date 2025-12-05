import { match, P } from "ts-pattern"

// Sigh, not sure how to get this automatically from Rust, yet. Vite seems to be getting in the way.
export type User = { user: string }
export type System = { system: string }
export type Assistant = { assistant: string }
export type Print = { print: string }
export type Seq = { seq: Query[] }
export type Par = { par: Query[] }
export type Plus = { plus: Query[] }
export type Cross = { cross: Query[] }
export type GenerateSpec = {
  model: string
  input: Query
  max_tokens: number
  temperature: number
}
export type Repeat = { repeat: { n: number; g: GenerateSpec } }
export type Generate = {
  g: GenerateSpec
}
export type Query =
  | Print
  | User
  | System
  | Assistant
  | Seq
  | Par
  | Plus
  | Cross
  | Repeat
  | Generate

export function isGenerate(u: Query): u is Generate {
  return match(u)
    .with({ g: P._ }, () => true)
    .otherwise(() => false)
}
