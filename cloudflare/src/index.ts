export default {
  async fetch(request: Request): Promise<Response> {
    return new Response("Hello Cloudflare!", {
      headers: { "Content-Type": "text/plain" },
    });
  },
} satisfies ExportedHandler;
