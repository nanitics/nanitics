"""Auction-routed request handling showcase runner.

Four specialist agents bid on every incoming request via
:class:`~nanitics.Bidding`; the winning :class:`~nanitics.ReActAgent`
answers. When no bid clears the confidence floor, the handler hands off
to a human via :class:`~nanitics.AsyncHumanInputProvider` against a
:class:`~nanitics.PostgresHitlRequestStore`.

See ``docker/full-stack/auction_routing/README.md`` for the pattern,
endpoint reference, and Observatory trace shape.
"""
