from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from graphql import (
    GraphQLEnumType,
    GraphQLField,
    GraphQLInterfaceType,
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLScalarType,
    GraphQLSchema,
    GraphQLUnionType,
    build_schema,
)


@dataclass(frozen=True)
class FieldArgument:
    name: str
    type: str


@dataclass(frozen=True)
class PathStep:
    type_name: str
    field_name: str
    return_type: str
    args: list[FieldArgument]


@dataclass(frozen=True)
class FieldNode:
    type_name: str
    field_name: str
    return_type: str
    args: list[FieldArgument]
    description: str | None
    summary: str
    search_text: str
    search_aliases: list[str]
    is_query_root: bool
    is_connection: bool
    is_scalar_return: bool
    is_list_return: bool
    is_reachable: bool
    coordinates: list[str] | None
    root_paths: list[list[dict[str, Any]]]


def describe_type(graphql_type) -> str:
    if isinstance(graphql_type, GraphQLNonNull):
        return f"{describe_type(graphql_type.of_type)}!"
    if isinstance(graphql_type, GraphQLList):
        return f"[{describe_type(graphql_type.of_type)}]"
    return str(graphql_type)


def unwrap_type(graphql_type):
    current = graphql_type
    while isinstance(current, (GraphQLNonNull, GraphQLList)):
        current = current.of_type
    return current


def is_list_type(graphql_type) -> bool:
    current = graphql_type
    while isinstance(current, GraphQLNonNull):
        current = current.of_type
    if isinstance(current, GraphQLList):
        return True
    if hasattr(current, "of_type"):
        return is_list_type(current.of_type)
    return False


def is_leaf_type(graphql_type) -> bool:
    unwrapped = unwrap_type(graphql_type)
    return isinstance(unwrapped, (GraphQLScalarType, GraphQLEnumType))


def unwrap_named_type_name(type_str: str) -> str:
    base = type_str.strip()
    while True:
        base = base.rstrip("!")
        if base.startswith("[") and base.endswith("]"):
            base = base[1:-1].strip()
            continue
        return base.rstrip("!")


def _normalize(text: str) -> str:
    chars: list[str] = []
    prev_space = False
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
            prev_space = False
        else:
            if not prev_space:
                chars.append(" ")
            prev_space = True
    return " ".join("".join(chars).split())


def tokenize(text: str) -> list[str]:
    normalized = _normalize(text)
    return [token for token in normalized.split(" ") if token]


def _format_args(args: list[FieldArgument], *, placeholders: bool) -> str:
    if not args:
        return ""
    rendered = []
    for arg in args:
        value = f"<{arg.type}>" if placeholders else arg.type
        rendered.append(f"{arg.name}: {value}")
    return f"({', '.join(rendered)})"


def _render_signature(type_name: str, field_name: str, args: list[FieldArgument], return_type: str) -> str:
    args_part = _format_args(args, placeholders=False)
    return f"{type_name}.{field_name}{args_part} -> {return_type}"


def _make_step(type_name: str, field_name: str, field: GraphQLField) -> PathStep:
    args = [
        FieldArgument(name=arg_name, type=describe_type(arg.type))
        for arg_name, arg in field.args.items()
    ]
    return PathStep(
        type_name=type_name,
        field_name=field_name,
        return_type=describe_type(field.type),
        args=args,
    )


def _field_to_target_type(field: GraphQLField) -> str | None:
    target = unwrap_type(field.type)
    if isinstance(target, (GraphQLObjectType, GraphQLInterfaceType, GraphQLUnionType)):
        return target.name
    return None


def _build_type_paths(schema: GraphQLSchema) -> dict[str, list[PathStep]]:
    query_type = schema.query_type
    if query_type is None:
        return {}

    visited: dict[str, list[PathStep]] = {query_type.name: []}
    queue: deque[str] = deque([query_type.name])

    while queue:
        type_name = queue.popleft()
        gql_type = schema.type_map.get(type_name)
        if not isinstance(gql_type, GraphQLObjectType):
            continue

        current_path = visited[type_name]
        for field_name, field in sorted(gql_type.fields.items()):
            target_name = _field_to_target_type(field)
            if not target_name or target_name in visited:
                continue
            visited[target_name] = [*current_path, _make_step(type_name, field_name, field)]
            queue.append(target_name)

    return visited


def _render_coordinates(
    path_steps: list[PathStep],
    target_type: str,
    target_field: str,
    target_args: list[FieldArgument],
) -> list[str] | None:
    if not path_steps and target_type != "Query":
        return None

    parts: list[str] = []
    for index, step in enumerate(path_steps):
        include_placeholders = index == 0 and step.type_name == "Query"
        parts.append(f"{step.type_name}.{step.field_name}{_format_args(step.args, placeholders=include_placeholders)}")
    if not path_steps or path_steps[-1].field_name != target_field or path_steps[-1].type_name != target_type:
        include_placeholders = target_type == "Query"
        parts.append(f"{target_type}.{target_field}{_format_args(target_args, placeholders=include_placeholders)}")
    return parts


def _build_search_aliases(
    type_name: str,
    field_name: str,
    return_type: str,
    coordinates: list[str] | None,
) -> list[str]:
    base_return = unwrap_named_type_name(return_type)
    aliases = [
        field_name,
        type_name,
        f"{type_name}.{field_name}",
        f"{type_name} {field_name}",
        return_type,
        base_return,
        f"{field_name} {base_return}",
    ]
    if coordinates:
        aliases.extend(coordinates)
        aliases.append(" -> ".join(coordinates))
    seen: set[str] = set()
    ordered: list[str] = []
    for alias in aliases:
        normalized = _normalize(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(alias)
    return ordered


def build_field_nodes(schema_text: str) -> list[FieldNode]:
    schema = build_schema(schema_text)
    type_paths = _build_type_paths(schema)
    nodes: list[FieldNode] = []

    for type_name, gql_type in sorted(schema.type_map.items()):
        if type_name.startswith("__"):
            continue
        if not isinstance(gql_type, GraphQLObjectType):
            continue

        root_path = type_paths.get(type_name)
        is_reachable_type = root_path is not None
        root_paths = [[asdict(step) for step in root_path]] if root_path is not None else []

        for field_name, field in sorted(gql_type.fields.items()):
            args = [
                FieldArgument(name=arg_name, type=describe_type(arg.type))
                for arg_name, arg in field.args.items()
            ]
            return_type = describe_type(field.type)
            summary = _render_signature(type_name, field_name, args, return_type)
            if field.description:
                summary = f"{summary} | desc: {field.description}"

            coordinates = _render_coordinates(root_path or [], type_name, field_name, args)
            aliases = _build_search_aliases(type_name, field_name, return_type, coordinates)
            description = field.description.strip() if field.description else None
            search_parts = [type_name, field_name, return_type, *aliases]
            if description:
                search_parts.append(description)
            if coordinates:
                search_parts.append(" -> ".join(coordinates))

            nodes.append(
                FieldNode(
                    type_name=type_name,
                    field_name=field_name,
                    return_type=return_type,
                    args=args,
                    description=description,
                    summary=summary,
                    search_text=" | ".join(search_parts),
                    search_aliases=aliases,
                    is_query_root=type_name == "Query",
                    is_connection=field_name.lower().endswith("connection"),
                    is_scalar_return=is_leaf_type(field.type),
                    is_list_return=is_list_type(field.type),
                    is_reachable=is_reachable_type,
                    coordinates=coordinates,
                    root_paths=root_paths,
                )
            )

    return nodes
