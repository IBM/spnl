import email from "../../../cli/src/builtins/email.yaml?raw"
import email2 from "../../../cli/src/builtins/email2.yaml?raw"
import email3 from "../../../cli/src/builtins/email3.yaml?raw"

export default [
  {
    value: "email",
    label: "Email Judge/Generator",
    description:
      "This demo is the simplest query, but does not generate great output",
    query: email
      .replace(/"\{n\}"/g, "4")
      .replace(/\{n\}/g, "4")
      .replace(/\{model\}/g, "model")
      .replace(/"\{temperature\}"/g, "0.2")
      .replace(/\{temperature\}/g, "0.2")
      .replace(/"\{max_tokens\}"/g, "100")
      .replace(/\{max_tokens\}/g, "100"),
  },

  {
    value: "email2",
    label: "Improved Email Judge/Generator",
    description:
      "This demo generates better output, at the expense of a more complicated query",
    query: email2
      .replace(/"\{n\}"/g, "4")
      .replace(/\{n\}/g, "4")
      .replace(/\{model\}/g, "model")
      .replace(/"\{temperature\}"/g, "0.2")
      .replace(/\{temperature\}/g, "0.2")
      .replace(/"\{max_tokens\}"/g, "100")
      .replace(/\{max_tokens\}/g, "100"),
  },

  {
    value: "email3",
    label: "Policy-driven Email Generation",
    description: "This demonstrates using policies to guide email generation",
    query: email3
      .replace(/"\{n\}"/g, "4")
      .replace(/\{n\}/g, "4")
      .replace(/\{model\}/g, "model")
      .replace(/"\{temperature\}"/g, "0.2")
      .replace(/\{temperature\}/g, "0.2")
      .replace(/"\{max_tokens\}"/g, "100")
      .replace(/\{max_tokens\}/g, "100"),
  },
]
